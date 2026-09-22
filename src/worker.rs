//! Worker command ordering, coalescing, and nonblocking UI-side lifetime.

use crate::log;
use crate::motion_tagging::{AuthoredOccurrenceRegistry, AuthoredTagRequest};
use crate::structure::{MainChannels, UserData};
use crate::utils::*;
use crate::wangtile::WangTile;
use std::sync::{
    Arc,
    atomic::{AtomicI32, Ordering},
    mpsc,
};

pub enum WorkerCommand {
    Configure(Box<UserData>),
    Build(bool, Vec3),
    View(Mat4),
    Membership(AuthoredTagRequest),
    Shutdown,
}

#[derive(Default)]
struct Wake {
    epoch: AtomicI32,
    #[cfg(not(target_arch = "wasm32"))]
    mutex: std::sync::Mutex<()>,
    #[cfg(not(target_arch = "wasm32"))]
    condition: std::sync::Condvar,
}

impl Wake {
    fn notify(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        let _guard = self.mutex.lock().unwrap_or_else(|error| error.into_inner());
        self.epoch.fetch_add(1, Ordering::Release);
        #[cfg(not(target_arch = "wasm32"))]
        self.condition.notify_one();
        #[cfg(target_arch = "wasm32")]
        // SAFETY: Arc keeps the naturally aligned atomic alive at a fixed address
        // in shared WASM memory. notify never waits and is legal on the UI thread.
        unsafe {
            core::arch::wasm32::memory_atomic_notify(self.epoch.as_ptr(), 1);
        }
    }

    fn wait(&self, observed: i32) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut guard = self.mutex.lock().unwrap_or_else(|error| error.into_inner());
            while self.epoch.load(Ordering::Acquire) == observed {
                guard = self
                    .condition
                    .wait(guard)
                    .unwrap_or_else(|error| error.into_inner());
            }
        }
        #[cfg(target_arch = "wasm32")]
        // SAFETY: called only by the worker, using the same live aligned atomic
        // as notify. The atomic comparison prevents lost wakes before this call.
        unsafe {
            core::arch::wasm32::memory_atomic_wait32(self.epoch.as_ptr(), observed, -1);
        }
    }
}

/// The queue never registers std::mpsc blocking waiters: its sender-side waker
/// mutex could otherwise make a browser main-thread send execute atomic.wait.
#[derive(Clone)]
pub struct WorkerSender {
    sender: Option<mpsc::Sender<WorkerCommand>>,
    wake: Arc<Wake>,
}

impl WorkerSender {
    pub fn send(&self, command: WorkerCommand) -> Result<(), mpsc::SendError<WorkerCommand>> {
        let result = self
            .sender
            .as_ref()
            .expect("live worker sender")
            .send(command);
        self.wake.notify();
        result
    }
}

impl Drop for WorkerSender {
    fn drop(&mut self) {
        // Disconnect the queue before waking a receiver that may be idle.
        drop(self.sender.take());
        self.wake.notify();
    }
}

struct WorkerInbox {
    receiver: mpsc::Receiver<WorkerCommand>,
    wake: Arc<Wake>,
}

impl WorkerInbox {
    fn recv(&self) -> Result<WorkerCommand, mpsc::RecvError> {
        loop {
            let observed = self.wake.epoch.load(Ordering::Acquire);
            match self.receiver.try_recv() {
                Ok(command) => return Ok(command),
                Err(mpsc::TryRecvError::Disconnected) => return Err(mpsc::RecvError),
                Err(mpsc::TryRecvError::Empty) => self.wake.wait(observed),
            }
        }
    }
}

fn worker_channel() -> (WorkerSender, WorkerInbox) {
    let (sender, receiver) = mpsc::channel();
    let wake = Arc::new(Wake::default());
    (
        WorkerSender {
            sender: Some(sender),
            wake: wake.clone(),
        },
        WorkerInbox { receiver, wake },
    )
}

#[derive(Default)]
struct WorkerBatch {
    config: Option<Box<UserData>>,
    build: Option<(bool, Vec3)>,
    view: Option<Mat4>,
    membership: Option<AuthoredTagRequest>,
}

fn receive_batch(inbox: &WorkerInbox) -> Option<WorkerBatch> {
    let mut command = inbox.recv().ok()?;
    let mut batch = WorkerBatch::default();
    loop {
        match command {
            WorkerCommand::Shutdown => return None,
            WorkerCommand::Configure(config) => {
                batch.config = Some(config);
                // Camera work queued before a configuration belongs to the old map.
                batch.build = None;
                batch.view = None;
            }
            WorkerCommand::Build(enabled, position) => batch.build = Some((enabled, position)),
            WorkerCommand::View(view) => batch.view = Some(view),
            WorkerCommand::Membership(request) => {
                if batch
                    .membership
                    .as_ref()
                    .is_none_or(|old| request.request_revision > old.request_revision)
                {
                    batch.membership = Some(request);
                }
            }
        }
        match inbox.receiver.try_recv() {
            Ok(next) => command = next,
            Err(mpsc::TryRecvError::Empty) => return Some(batch),
            Err(mpsc::TryRecvError::Disconnected) => return None,
        }
    }
}

/// Owning this handle owns worker lifetime. Drop signals shutdown, never joins:
/// browser main threads must remain free to service the worker and the event loop.
pub struct WorkerHandle {
    sender: WorkerSender,
    thread: Option<wasm_thread::JoinHandle<()>>,
}

impl WorkerHandle {
    fn spawn(
        mut handle_batch: impl FnMut(WorkerBatch) -> bool + Send + 'static,
    ) -> (WorkerSender, Self) {
        let (sender, inbox) = worker_channel();
        let thread = wasm_thread::spawn(move || {
            while let Some(batch) = receive_batch(&inbox) {
                if !handle_batch(batch) {
                    break;
                }
            }
        });
        (
            sender.clone(),
            Self {
                sender,
                thread: Some(thread),
            },
        )
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        let _ = self.sender.send(WorkerCommand::Shutdown);
        drop(self.thread.take());
    }
}

pub fn launch_worker_thread(mut wang: WangTile) -> (MainChannels, WorkerHandle) {
    let (tx_user_data, rx_user_data) = mpsc::channel();
    let (tx_sort_data, rx_sort_data) = mpsc::channel();
    let (tx_scene_data, rx_scene_data) = mpsc::channel();
    let (tx_sort_time, rx_sort_time) = mpsc::channel();
    let (tx_build_time, rx_build_time) = mpsc::channel();
    let mut camera_pos = None;
    let mut previous_view = None;
    let mut current_view = None;
    let mut next_scene_id = 0u32;
    let mut registry = AuthoredOccurrenceRegistry::default();
    let mut force_sort = false;
    let (tx_commands, handle) = WorkerHandle::spawn(move |batch| {
        if let Some(config) = batch.config {
            if tx_user_data.send(wang.configure(*config)).is_err() {
                return false;
            }
            camera_pos = None;
            previous_view = None;
            current_view = None;
        }
        if let Some(request) = batch.membership
            && request.request_revision > registry.request_revision()
        {
            match registry.reset(next_scene_id.saturating_sub(1), request) {
                Ok(()) => force_sort = true,
                Err(error) => {
                    log!("Rejected authored membership request: {error}");
                }
            }
        }
        if let Some((do_build, position)) = batch.build {
            camera_pos = Some(position);
            if do_build && wang.check_update(&position) {
                let start = get_time_milliseconds();
                let mut scene = wang.build_tiles(position);
                scene.scene_id = next_scene_id;
                if let Err(error) = registry.reset(next_scene_id, registry.current_request()) {
                    log!("Failed to retarget authored occurrence registry: {error}");
                    return false;
                }
                if tx_scene_data.send(scene).is_err() {
                    return false;
                }
                let _ = tx_build_time.send(get_time_milliseconds() - start);
                next_scene_id += 1;
                // A newly built map needs a matching sort even with a locked or
                // numerically unchanged view; reuse the last available view.
                force_sort = true;
            }
        }
        let received_view = batch.view.is_some();
        if received_view {
            current_view = batch.view;
        }
        if let Some(position) = camera_pos
            && should_worker_sort(
                force_sort,
                received_view,
                wang.user_data.always_sort,
                previous_view,
                current_view,
            )
        {
            let view = current_view.expect("sort decision requires a current view");
            let start = get_time_milliseconds();
            match wang.sort_tiles(position, view, &mut registry) {
                Ok(mut sorted) => {
                    sorted.scene_id = next_scene_id.saturating_sub(1);
                    previous_view = Some(view);
                    if tx_sort_data.send(sorted).is_err() {
                        return false;
                    }
                    let _ = tx_sort_time.send(get_time_milliseconds() - start);
                }
                Err(error) => {
                    log!("Rejected authored worker sort: {error}");
                }
            }
            force_sort = false;
        }
        true
    });
    (
        MainChannels {
            tx_commands,
            rx_user_data,
            rx_sort_data,
            rx_scene_data,
            rx_sort_time,
            rx_build_time,
            rx_fly_path_control: None,
            rx_height_tex: None,
            rx_skybox_tex: None,
            rx_proxy_tex: None,
        },
        handle,
    )
}

fn view_matrix_difference(previous: Mat4, current: Mat4) -> f32 {
    let diff = previous - current;
    diff[0][0].abs()
        + diff[0][1].abs()
        + diff[0][2].abs()
        + diff[0][3].abs()
        + diff[1][0].abs()
        + diff[1][1].abs()
        + diff[1][2].abs()
        + diff[1][3].abs()
        + diff[2][0].abs()
        + diff[2][1].abs()
        + diff[2][2].abs()
        + diff[2][3].abs()
        + diff[3][0].abs()
        + diff[3][1].abs()
        + diff[3][2].abs()
        + diff[3][3].abs()
}

pub(crate) fn should_worker_sort(
    force_sort: bool,
    received_view: bool,
    always_sort: bool,
    previous_view: Option<Mat4>,
    current_view: Option<Mat4>,
) -> bool {
    let Some(current_view) = current_view else {
        return false;
    };
    if force_sort {
        return true;
    }
    if !received_view {
        return false;
    }
    always_sort
        || previous_view
            .is_none_or(|previous| view_matrix_difference(previous, current_view) >= 0.01)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn inbox_coalesces_motion_and_camera_but_configuration_is_a_barrier() {
        let (tx, rx) = worker_channel();
        tx.send(WorkerCommand::View(Mat4::from_scale(1.0))).unwrap();
        tx.send(WorkerCommand::Build(true, Vec3::new(1.0, 0.0, 0.0)))
            .unwrap();
        tx.send(WorkerCommand::Configure(Box::new(UserData::new())))
            .unwrap();
        for revision in [3, 1, 2] {
            tx.send(WorkerCommand::Membership(AuthoredTagRequest {
                request_revision: revision,
                snapshot: None,
            }))
            .unwrap();
        }
        tx.send(WorkerCommand::View(Mat4::from_scale(2.0))).unwrap();
        tx.send(WorkerCommand::View(Mat4::from_scale(3.0))).unwrap();
        let batch = receive_batch(&rx).unwrap();
        assert!(batch.config.is_some());
        assert!(batch.build.is_none());
        assert_eq!(batch.view, Some(Mat4::from_scale(3.0)));
        assert_eq!(batch.membership.unwrap().request_revision, 3);
    }

    #[test]
    fn inbox_waits_for_input_then_exits_on_disconnect() {
        let (tx, rx) = worker_channel();
        let (result_tx, result_rx) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            result_tx.send(receive_batch(&rx).map(|b| b.view)).unwrap();
            result_tx.send(receive_batch(&rx).map(|b| b.view)).unwrap();
        });
        assert!(matches!(
            result_rx.recv_timeout(Duration::from_millis(30)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        tx.send(WorkerCommand::View(Mat4::from_scale(2.0))).unwrap();
        assert_eq!(
            result_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            Some(Some(Mat4::from_scale(2.0)))
        );
        drop(tx);
        assert_eq!(
            result_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            None
        );
        thread.join().unwrap();
    }

    #[test]
    fn shutdown_discards_queued_work() {
        let (tx, rx) = worker_channel();
        tx.send(WorkerCommand::View(Mat4::from_scale(1.0))).unwrap();
        tx.send(WorkerCommand::Shutdown).unwrap();
        assert!(receive_batch(&rx).is_none());
    }

    struct ExitNotice(mpsc::Sender<()>);
    impl Drop for ExitNotice {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }

    #[test]
    fn dropping_worker_wakes_it_without_joining_the_ui_thread() {
        let (done_tx, done_rx) = mpsc::channel();
        let notice = ExitNotice(done_tx);
        let (tx, handle) = WorkerHandle::spawn(move |_| {
            let _ = &notice;
            true
        });
        drop(handle);
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(tx.send(WorkerCommand::View(Mat4::from_scale(1.0))).is_err());
    }

    #[test]
    fn result_consumer_disconnect_can_end_the_worker() {
        let (done_tx, done_rx) = mpsc::channel();
        let notice = ExitNotice(done_tx);
        let (tx, handle) = WorkerHandle::spawn(move |_| {
            let _ = &notice;
            false
        });
        tx.send(WorkerCommand::View(Mat4::from_scale(1.0))).unwrap();
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        drop(handle);
    }
}
