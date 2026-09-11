use std::{
    collections::{BTreeSet, VecDeque},
    sync::mpsc::{self, Receiver, Sender},
};

const GPU_QUERY_COUNT: u32 = 6;
const GPU_TIMESTAMP_BYTES: u64 = GPU_QUERY_COUNT as u64 * std::mem::size_of::<u64>() as u64;
const GPU_TIMESTAMP_PAIR_BYTES: u64 = 2 * std::mem::size_of::<u64>() as u64;
const MOTION_RESOLVE_OFFSET: u64 = 0;
const AUTHORED_RESOLVE_OFFSET: u64 = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;
const RENDER_RESOLVE_OFFSET: u64 = 2 * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;
const GPU_RESOLVE_BUFFER_BYTES: u64 = RENDER_RESOLVE_OFFSET + GPU_TIMESTAMP_PAIR_BYTES;
const READBACK_RING_SIZE: usize = 4;

pub fn renderer_required_features(adapter_features: wgpu::Features) -> wgpu::Features {
    let mut required = wgpu::Features::FLOAT32_FILTERABLE;
    if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required |= wgpu::Features::TIMESTAMP_QUERY;
    }
    required
}

#[derive(Debug)]
pub struct ReadbackSlots {
    busy: Vec<bool>,
    next: usize,
}

impl ReadbackSlots {
    pub fn new(count: usize) -> Self {
        assert!(count > 0, "readback ring must not be empty");
        Self {
            busy: vec![false; count],
            next: 0,
        }
    }

    pub fn acquire(&mut self) -> Option<usize> {
        for offset in 0..self.busy.len() {
            let index = (self.next + offset) % self.busy.len();
            if !self.busy[index] {
                self.busy[index] = true;
                self.next = (index + 1) % self.busy.len();
                return Some(index);
            }
        }
        None
    }

    pub fn release(&mut self, index: usize) {
        if let Some(busy) = self.busy.get_mut(index) {
            *busy = false;
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MetricSummary {
    pub mean: f64,
    pub p95: f64,
    pub samples: usize,
}

#[derive(Debug)]
pub struct MetricSeries {
    values: VecDeque<f64>,
    capacity: usize,
}

impl MetricSeries {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "metric window must be positive");
        Self {
            values: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn add(&mut self, value: f64) {
        if !value.is_finite() || value < 0.0 {
            return;
        }
        if self.values.len() == self.capacity {
            self.values.pop_front();
        }
        self.values.push_back(value);
    }

    pub fn summary(&self) -> MetricSummary {
        if self.values.is_empty() {
            return MetricSummary::default();
        }
        let mean = self.values.iter().sum::<f64>() / self.values.len() as f64;
        let mut sorted = self.values.iter().copied().collect::<Vec<_>>();
        sorted.sort_by(f64::total_cmp);
        let rank = ((0.95 * sorted.len() as f64).ceil() as usize).max(1);
        MetricSummary {
            mean,
            p95: sorted[rank - 1],
            samples: sorted.len(),
        }
    }
}

pub fn duration_milliseconds(start: u64, end: u64, period_nanoseconds: f32) -> Option<f64> {
    if end < start || !period_nanoseconds.is_finite() || period_nanoseconds <= 0.0 {
        return None;
    }
    Some((end - start) as f64 * period_nanoseconds as f64 / 1_000_000.0)
}

#[derive(Clone, Debug, Default)]
pub struct FrameCounters {
    pub archive_rows: usize,
    pub motion_rows: usize,
    pub selected_splats: usize,
    pub rendered_splats: usize,
    pub blending_splats: usize,
    pub draw_calls: usize,
    pub uploaded_bytes: u64,
    pub active_members: usize,
    pub total_members: usize,
    pub motion_dispatched: bool,
    pub motion_blend: bool,
    pub authored_registry_occurrences: usize,
    pub authored_tagged_occurrences: usize,
    pub authored_affected_draws: usize,
    pub authored_membership_request_revision: u64,
    pub authored_registry_revision: u64,
    pub authored_worker_tag_ms: f64,
    pub authored_registry_installed: bool,
    pub authored_registry_upload_bytes: u64,
    pub authored_dispatched: bool,
    active_member_set: BTreeSet<(usize, usize)>,
}

impl FrameCounters {
    pub fn for_render(selected_splats: usize, blending_splats: usize, uploaded_bytes: u64) -> Self {
        Self {
            selected_splats,
            blending_splats,
            uploaded_bytes,
            ..Self::default()
        }
    }

    pub fn record_draw<I>(&mut self, splats: usize, uploaded_bytes: u64, members: I)
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        self.draw_calls += 1;
        self.rendered_splats += splats;
        self.uploaded_bytes = self.uploaded_bytes.saturating_add(uploaded_bytes);
        self.active_member_set.extend(members);
        self.active_members = self.active_member_set.len();
    }

    pub fn merge_render_work(&mut self, work: FrameCounters) {
        self.rendered_splats = work.rendered_splats;
        self.selected_splats = work.selected_splats;
        self.blending_splats = work.blending_splats;
        self.draw_calls = work.draw_calls;
        self.uploaded_bytes = self.uploaded_bytes.saturating_add(work.uploaded_bytes);
        self.active_member_set.extend(work.active_member_set);
        self.active_members = self.active_member_set.len();
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionDispatchWork {
    pub rows: usize,
    pub uploaded_bytes: u64,
    pub blend: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuTimingSample {
    pub motion: Option<(u64, u64)>,
    pub authored: Option<(u64, u64)>,
    pub render: Option<(u64, u64)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TimestampPasses {
    motion: bool,
    authored: bool,
    render: bool,
}

fn decode_timestamp_words(
    words: [u64; GPU_QUERY_COUNT as usize],
    passes: TimestampPasses,
) -> GpuTimingSample {
    GpuTimingSample {
        motion: passes.motion.then_some((words[0], words[1])),
        authored: passes.authored.then_some((words[2], words[3])),
        render: passes.render.then_some((words[4], words[5])),
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProfileSnapshot {
    pub enabled: bool,
    pub frame_cpu: MetricSummary,
    pub motion_cpu: MetricSummary,
    pub render_cpu: MetricSummary,
    pub worker_sort_cpu: MetricSummary,
    pub worker_build_cpu: MetricSummary,
    pub motion_gpu: MetricSummary,
    pub authored_gpu: MetricSummary,
    pub gaussian_gpu: MetricSummary,
    pub counters: FrameCounters,
    pub gpu_timestamps_supported: bool,
    pub gpu_error: Option<String>,
}

#[derive(Debug)]
pub struct ProfileHistory {
    frame_cpu: MetricSeries,
    motion_cpu: MetricSeries,
    render_cpu: MetricSeries,
    worker_sort_cpu: MetricSeries,
    worker_build_cpu: MetricSeries,
    motion_gpu: MetricSeries,
    authored_gpu: MetricSeries,
    gaussian_gpu: MetricSeries,
}

#[derive(Debug)]
struct ReadbackResult {
    slot: usize,
    sample: Result<GpuTimingSample, String>,
}

#[derive(Clone, Copy, Debug)]
struct PendingReadback {
    slot: usize,
    passes: TimestampPasses,
}

#[derive(Debug)]
struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffers: Vec<wgpu::Buffer>,
    slots: ReadbackSlots,
    current_slot: Option<usize>,
    current_passes: TimestampPasses,
    pending_readback: Option<PendingReadback>,
    timestamp_period: f32,
    sender: Sender<ReadbackResult>,
    receiver: Receiver<ReadbackResult>,
    last_error: Option<String>,
}

impl GpuProfiler {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Renderer profiler timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: GPU_QUERY_COUNT,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer profiler query resolve"),
            size: GPU_RESOLVE_BUFFER_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffers = (0..READBACK_RING_SIZE)
            .map(|index| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Renderer profiler readback {index}")),
                    size: GPU_TIMESTAMP_BYTES,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let (sender, receiver) = mpsc::channel();
        Self {
            query_set,
            resolve_buffer,
            readback_buffers,
            slots: ReadbackSlots::new(READBACK_RING_SIZE),
            current_slot: None,
            current_passes: TimestampPasses::default(),
            pending_readback: None,
            timestamp_period: queue.get_timestamp_period(),
            sender,
            receiver,
            last_error: None,
        }
    }

    fn drain_completed(&mut self, history: &mut ProfileHistory) {
        while let Ok(result) = self.receiver.try_recv() {
            self.slots.release(result.slot);
            match result.sample {
                Ok(sample) => {
                    history.record_gpu(sample, self.timestamp_period);
                    self.last_error = None;
                }
                Err(error) => self.last_error = Some(error),
            }
        }
    }

    fn begin_frame(&mut self, enabled: bool) {
        if let Some(abandoned_slot) = self.current_slot.take() {
            self.slots.release(abandoned_slot);
        }
        self.current_passes = TimestampPasses::default();
        self.pending_readback = None;
        self.current_slot = enabled.then(|| self.slots.acquire()).flatten();
    }

    fn motion_timestamp_writes(&mut self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.current_slot?;
        self.current_passes.motion = true;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        })
    }

    fn render_timestamp_writes(&mut self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.current_slot?;
        self.current_passes.render = true;
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(4),
            end_of_pass_write_index: Some(5),
        })
    }

    fn authored_timestamp_writes(&mut self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.current_slot?;
        self.current_passes.authored = true;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(2),
            end_of_pass_write_index: Some(3),
        })
    }

    fn cancel_motion_timestamp(&mut self) {
        self.current_passes.motion = false;
    }

    fn cancel_authored_timestamp(&mut self) {
        self.current_passes.authored = false;
    }

    fn finish_encoding(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(slot) = self.current_slot.take() else {
            return;
        };
        if !self.current_passes.motion
            && !self.current_passes.authored
            && !self.current_passes.render
        {
            self.slots.release(slot);
            return;
        }
        if self.current_passes.motion {
            encoder.resolve_query_set(
                &self.query_set,
                0..2,
                &self.resolve_buffer,
                MOTION_RESOLVE_OFFSET,
            );
            encoder.copy_buffer_to_buffer(
                &self.resolve_buffer,
                MOTION_RESOLVE_OFFSET,
                &self.readback_buffers[slot],
                0,
                GPU_TIMESTAMP_PAIR_BYTES,
            );
        }
        if self.current_passes.render {
            encoder.resolve_query_set(
                &self.query_set,
                4..6,
                &self.resolve_buffer,
                RENDER_RESOLVE_OFFSET,
            );
            encoder.copy_buffer_to_buffer(
                &self.resolve_buffer,
                RENDER_RESOLVE_OFFSET,
                &self.readback_buffers[slot],
                2 * GPU_TIMESTAMP_PAIR_BYTES,
                GPU_TIMESTAMP_PAIR_BYTES,
            );
        }
        if self.current_passes.authored {
            encoder.resolve_query_set(
                &self.query_set,
                2..4,
                &self.resolve_buffer,
                AUTHORED_RESOLVE_OFFSET,
            );
            encoder.copy_buffer_to_buffer(
                &self.resolve_buffer,
                AUTHORED_RESOLVE_OFFSET,
                &self.readback_buffers[slot],
                GPU_TIMESTAMP_PAIR_BYTES,
                GPU_TIMESTAMP_PAIR_BYTES,
            );
        }
        self.pending_readback = Some(PendingReadback {
            slot,
            passes: self.current_passes,
        });
    }

    fn schedule_readback(&mut self, command_buffer: &wgpu::CommandBuffer) {
        let Some(pending) = self.pending_readback.take() else {
            return;
        };
        let mapping_buffer = self.readback_buffers[pending.slot].clone();
        let callback_buffer = mapping_buffer.clone();
        let sender = self.sender.clone();
        command_buffer.map_buffer_on_submit(
            &mapping_buffer,
            wgpu::MapMode::Read,
            ..,
            move |result| {
                let sample = result.map_err(|error| error.to_string()).and_then(|()| {
                    let view = callback_buffer.slice(..).get_mapped_range();
                    if view.len() != GPU_TIMESTAMP_BYTES as usize {
                        drop(view);
                        callback_buffer.unmap();
                        return Err("timestamp readback had an unexpected size".to_string());
                    }
                    let mut words = [0_u64; GPU_QUERY_COUNT as usize];
                    for (word, bytes) in words.iter_mut().zip(view.chunks_exact(8)) {
                        *word = u64::from_ne_bytes(bytes.try_into().expect("timestamp chunk size"));
                    }
                    drop(view);
                    callback_buffer.unmap();
                    Ok(decode_timestamp_words(words, pending.passes))
                });
                let _ = sender.send(ReadbackResult {
                    slot: pending.slot,
                    sample,
                });
            },
        );
    }
}

#[derive(Debug)]
pub struct FrameProfiler {
    enabled: bool,
    history: ProfileHistory,
    counters: FrameCounters,
    gpu: Option<GpuProfiler>,
}

impl FrameProfiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, history_window: usize) -> Self {
        let gpu = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| GpuProfiler::new(device, queue));
        Self {
            enabled: true,
            history: ProfileHistory::new(history_window),
            counters: FrameCounters::default(),
            gpu,
        }
    }

    pub fn begin_frame(&mut self, device: &wgpu::Device, enabled: bool) {
        self.enabled = enabled;
        self.counters = FrameCounters::default();
        if let Some(gpu) = self.gpu.as_mut() {
            let _ = device.poll(wgpu::PollType::Poll);
            gpu.drain_completed(&mut self.history);
            gpu.begin_frame(enabled);
        }
    }

    pub fn motion_timestamp_writes(&mut self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.enabled
            .then(|| self.gpu.as_mut()?.motion_timestamp_writes())
            .flatten()
    }

    pub fn render_timestamp_writes(&mut self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.enabled
            .then(|| self.gpu.as_mut()?.render_timestamp_writes())
            .flatten()
    }

    pub fn authored_timestamp_writes(&mut self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.enabled
            .then(|| self.gpu.as_mut()?.authored_timestamp_writes())
            .flatten()
    }

    pub fn cancel_motion_timestamp(&mut self) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.cancel_motion_timestamp();
        }
    }

    pub fn cancel_authored_timestamp(&mut self) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.cancel_authored_timestamp();
        }
    }

    pub fn finish_encoding(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.finish_encoding(encoder);
        }
    }

    pub fn schedule_readback(&mut self, command_buffer: &wgpu::CommandBuffer) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.schedule_readback(command_buffer);
        }
    }

    pub fn record_frame_cpu(&mut self, milliseconds: f64) {
        if self.enabled {
            self.history.record_frame_cpu(milliseconds);
        }
    }

    pub fn record_motion_cpu(&mut self, milliseconds: f64) {
        if self.enabled {
            self.history.record_motion_cpu(milliseconds);
        }
    }

    pub fn record_render_cpu(&mut self, milliseconds: f64) {
        if self.enabled {
            self.history.record_render_cpu(milliseconds);
        }
    }

    pub fn record_worker_sort_cpu(&mut self, milliseconds: f64) {
        if self.enabled {
            self.history.record_worker_sort_cpu(milliseconds);
        }
    }

    pub fn record_worker_build_cpu(&mut self, milliseconds: f64) {
        if self.enabled {
            self.history.record_worker_build_cpu(milliseconds);
        }
    }

    pub fn counters_mut(&mut self) -> &mut FrameCounters {
        &mut self.counters
    }

    pub fn snapshot(&self) -> ProfileSnapshot {
        let mut snapshot = self.history.snapshot();
        snapshot.enabled = self.enabled;
        snapshot.counters = self.counters.clone();
        snapshot.gpu_timestamps_supported = self.gpu.is_some();
        snapshot.gpu_error = self.gpu.as_ref().and_then(|gpu| gpu.last_error.clone());
        snapshot
    }

    pub fn clear(&mut self) {
        self.history.clear();
    }
}

impl ProfileHistory {
    pub fn new(window: usize) -> Self {
        Self {
            frame_cpu: MetricSeries::new(window),
            motion_cpu: MetricSeries::new(window),
            render_cpu: MetricSeries::new(window),
            worker_sort_cpu: MetricSeries::new(window),
            worker_build_cpu: MetricSeries::new(window),
            motion_gpu: MetricSeries::new(window),
            authored_gpu: MetricSeries::new(window),
            gaussian_gpu: MetricSeries::new(window),
        }
    }

    pub fn record_frame_cpu(&mut self, milliseconds: f64) {
        self.frame_cpu.add(milliseconds);
    }

    pub fn record_motion_cpu(&mut self, milliseconds: f64) {
        self.motion_cpu.add(milliseconds);
    }

    pub fn record_render_cpu(&mut self, milliseconds: f64) {
        self.render_cpu.add(milliseconds);
    }

    pub fn record_worker_sort_cpu(&mut self, milliseconds: f64) {
        self.worker_sort_cpu.add(milliseconds);
    }

    pub fn record_worker_build_cpu(&mut self, milliseconds: f64) {
        self.worker_build_cpu.add(milliseconds);
    }

    pub fn record_gpu(&mut self, sample: GpuTimingSample, period_nanoseconds: f32) {
        if let Some((start, end)) = sample.motion {
            if let Some(milliseconds) = duration_milliseconds(start, end, period_nanoseconds) {
                self.motion_gpu.add(milliseconds);
            }
        }
        if let Some((start, end)) = sample.render {
            if let Some(milliseconds) = duration_milliseconds(start, end, period_nanoseconds) {
                self.gaussian_gpu.add(milliseconds);
            }
        }
        if let Some((start, end)) = sample.authored {
            if let Some(milliseconds) = duration_milliseconds(start, end, period_nanoseconds) {
                self.authored_gpu.add(milliseconds);
            }
        }
    }

    pub fn snapshot(&self) -> ProfileSnapshot {
        ProfileSnapshot {
            frame_cpu: self.frame_cpu.summary(),
            motion_cpu: self.motion_cpu.summary(),
            render_cpu: self.render_cpu.summary(),
            worker_sort_cpu: self.worker_sort_cpu.summary(),
            worker_build_cpu: self.worker_build_cpu.summary(),
            motion_gpu: self.motion_gpu.summary(),
            authored_gpu: self.authored_gpu.summary(),
            gaussian_gpu: self.gaussian_gpu.summary(),
            ..ProfileSnapshot::default()
        }
    }

    pub fn clear(&mut self) {
        for series in [
            &mut self.frame_cpu,
            &mut self.motion_cpu,
            &mut self.render_cpu,
            &mut self.worker_sort_cpu,
            &mut self.worker_build_cpu,
            &mut self.motion_gpu,
            &mut self.authored_gpu,
            &mut self.gaussian_gpu,
        ] {
            series.values.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FrameCounters, GpuTimingSample, MetricSeries, ProfileHistory, ReadbackSlots,
        TimestampPasses, decode_timestamp_words, duration_milliseconds, renderer_required_features,
    };

    #[test]
    fn rolling_metrics_report_mean_and_nearest_rank_p95() {
        let mut series = MetricSeries::new(4);
        for value in [1.0, 2.0, 3.0, 100.0, 5.0] {
            series.add(value);
        }

        let summary = series.summary();
        assert_eq!(summary.samples, 4);
        assert!((summary.mean - 27.5).abs() < 1.0e-12);
        assert_eq!(summary.p95, 100.0);
    }

    #[test]
    fn rolling_metrics_ignore_negative_and_nonfinite_samples() {
        let mut series = MetricSeries::new(4);
        series.add(2.0);
        series.add(-1.0);
        series.add(f64::NAN);
        series.add(f64::INFINITY);

        let summary = series.summary();
        assert_eq!(summary.samples, 1);
        assert_eq!(summary.mean, 2.0);
        assert_eq!(summary.p95, 2.0);
    }

    #[test]
    fn timestamp_ticks_convert_to_milliseconds_without_underflow() {
        assert_eq!(duration_milliseconds(100, 350, 2.0), Some(0.0005));
        assert_eq!(duration_milliseconds(350, 100, 2.0), None);
        assert_eq!(duration_milliseconds(100, 350, 0.0), None);
    }

    #[test]
    fn frame_counters_accumulate_draw_and_upload_work() {
        let mut counters = FrameCounters::default();
        counters.record_draw(20, 64, [(0, 3), (1, 3)]);
        counters.record_draw(12, 32, [(0, 3), (2, 5)]);

        assert_eq!(counters.draw_calls, 2);
        assert_eq!(counters.rendered_splats, 32);
        assert_eq!(counters.uploaded_bytes, 96);
        assert_eq!(counters.active_members, 3);
    }

    #[test]
    fn render_work_merge_preserves_motion_uploads_and_adds_selection_counts() {
        let mut frame = FrameCounters {
            motion_rows: 100,
            authored_registry_occurrences: 12,
            authored_tagged_occurrences: 75,
            authored_affected_draws: 3,
            authored_membership_request_revision: 4,
            authored_registry_revision: 9,
            authored_worker_tag_ms: 0.6,
            authored_registry_installed: true,
            authored_registry_upload_bytes: 512,
            authored_dispatched: true,
            uploaded_bytes: 40,
            ..FrameCounters::default()
        };
        let mut render = FrameCounters::for_render(90, 15, 20);
        render.record_draw(80, 30, [(0, 0)]);

        frame.merge_render_work(render);

        assert_eq!(frame.motion_rows, 100);
        assert_eq!(frame.authored_registry_occurrences, 12);
        assert_eq!(frame.authored_tagged_occurrences, 75);
        assert_eq!(frame.authored_affected_draws, 3);
        assert_eq!(frame.authored_membership_request_revision, 4);
        assert_eq!(frame.authored_registry_revision, 9);
        assert_eq!(frame.authored_worker_tag_ms, 0.6);
        assert!(frame.authored_registry_installed);
        assert_eq!(frame.authored_registry_upload_bytes, 512);
        assert!(frame.authored_dispatched);
        assert_eq!(frame.selected_splats, 90);
        assert_eq!(frame.rendered_splats, 80);
        assert_eq!(frame.blending_splats, 15);
        assert_eq!(frame.uploaded_bytes, 90);
    }

    #[test]
    fn profile_history_keeps_cpu_and_gpu_passes_separate() {
        let mut history = ProfileHistory::new(8);
        history.record_frame_cpu(6.0);
        history.record_motion_cpu(1.5);
        history.record_render_cpu(2.5);
        history.record_gpu(
            GpuTimingSample {
                motion: Some((100, 600)),
                authored: Some((700, 1200)),
                render: Some((1300, 2300)),
            },
            2.0,
        );

        let snapshot = history.snapshot();
        assert_eq!(snapshot.frame_cpu.mean, 6.0);
        assert_eq!(snapshot.motion_cpu.mean, 1.5);
        assert_eq!(snapshot.render_cpu.mean, 2.5);
        assert_eq!(snapshot.motion_gpu.mean, 0.001);
        assert_eq!(snapshot.authored_gpu.mean, 0.001);
        assert_eq!(snapshot.gaussian_gpu.mean, 0.002);
    }

    #[test]
    fn absent_gpu_pass_does_not_add_a_zero_duration_sample() {
        let mut history = ProfileHistory::new(8);
        history.record_gpu(
            GpuTimingSample {
                motion: None,
                authored: None,
                render: Some((5, 10)),
            },
            1.0,
        );

        let snapshot = history.snapshot();
        assert_eq!(snapshot.motion_gpu.samples, 0);
        assert_eq!(snapshot.authored_gpu.samples, 0);
        assert_eq!(snapshot.gaussian_gpu.samples, 1);
    }

    #[test]
    fn timestamp_feature_is_requested_only_when_the_adapter_supports_it() {
        let base = renderer_required_features(wgpu::Features::empty());
        assert!(base.contains(wgpu::Features::FLOAT32_FILTERABLE));
        assert!(!base.contains(wgpu::Features::TIMESTAMP_QUERY));

        let timestamped = renderer_required_features(wgpu::Features::TIMESTAMP_QUERY);
        assert!(timestamped.contains(wgpu::Features::FLOAT32_FILTERABLE));
        assert!(timestamped.contains(wgpu::Features::TIMESTAMP_QUERY));
    }

    #[test]
    fn readback_ring_skips_instead_of_reusing_busy_buffers() {
        let mut slots = ReadbackSlots::new(2);
        assert_eq!(slots.acquire(), Some(0));
        assert_eq!(slots.acquire(), Some(1));
        assert_eq!(slots.acquire(), None);
        slots.release(0);
        assert_eq!(slots.acquire(), Some(0));
    }

    #[test]
    fn timestamp_decoder_omits_passes_that_were_not_written() {
        let sample = decode_timestamp_words(
            [10, 20, 30, 50, 80, 120],
            TimestampPasses {
                motion: true,
                authored: true,
                render: false,
            },
        );
        assert_eq!(sample.motion, Some((10, 20)));
        assert_eq!(sample.authored, Some((30, 50)));
        assert_eq!(sample.render, None);
    }

    #[test]
    fn timestamp_decoder_keeps_global_authored_and_render_pairs_independent() {
        let sample = decode_timestamp_words(
            [10, 20, 30, 50, 80, 120],
            TimestampPasses {
                motion: true,
                authored: true,
                render: true,
            },
        );

        assert_eq!(sample.motion, Some((10, 20)));
        assert_eq!(sample.authored, Some((30, 50)));
        assert_eq!(sample.render, Some((80, 120)));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn profiler_query_resolve_layout_passes_webgpu_validation() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })) {
                Ok(adapter) => adapter,
                Err(error) => {
                    eprintln!("timestamp layout test skipped: no native adapter ({error})");
                    return;
                }
            };
        if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            eprintln!("timestamp layout test skipped: adapter lacks timestamp queries");
            return;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("timestamp layout test device"),
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .unwrap();
        let mut profiler = super::GpuProfiler::new(&device, &queue);
        profiler.begin_frame(true);
        assert!(profiler.motion_timestamp_writes().is_some());
        assert!(profiler.authored_timestamp_writes().is_some());
        assert!(profiler.render_timestamp_writes().is_some());

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("timestamp layout test encoder"),
        });
        profiler.finish_encoding(&mut encoder);
        let _ = encoder.finish();
        let validation_error = pollster::block_on(device.pop_error_scope());

        assert!(
            validation_error.is_none(),
            "profiler query resolve layout must be WebGPU-valid: {validation_error:?}"
        );
    }
}
