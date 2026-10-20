"""Local model inventory and provenance; no hardware access or model downloads."""
from importlib import metadata
from pathlib import Path

from childsafe.evaluation.sessions import code_hash, config_hash, file_hash
from childsafe.product import product_scope
from childsafe.vision.detection import describe_selection, select_weights


def demo_readiness(cfg, checkpoint):
    models = {}
    for role, value in (('classifier', checkpoint), ('detector', select_weights(cfg['detector'])),
                        ('face', cfg['paths']['face_model']), ('hand', cfg['paths']['hand_model'])):
        path = Path(value)
        exists = path.is_file()
        models[role] = dict(file=path.name, exists=exists,
                            sha256=file_hash(path) if exists else None,
                            bytes=path.stat().st_size if exists else None)
    versions = {}
    for name in ('torch','torchvision','mediapipe','ultralytics','opencv-python','numpy'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    missing = [role for role, model in models.items() if not model['exists']]
    return dict(status='missing_local_models' if missing else 'local_files_present_not_camera_validated',
                missing_models=missing, models=models, versions=versions,
                detector=describe_selection(cfg['detector']), config_sha256=config_hash(cfg),
                code_sha256=code_hash(), scope=product_scope(cfg),
                camera_opened=False, video_saved=False,
                camera_event_recall=None, false_alarms_per_hour=None, camera_latency_s=None,
                remaining=['Kiểm tra nạp model và suy luận bằng smoke test trên ảnh không có người.',
                           'Thu phiên người lớn/ma-nơ-canh có nhãn độc lập và quy trình được trường xác nhận.',
                           'Đo lại sau sửa trên camera thật; benchmark tổng hợp không thay thế bước này.'])
