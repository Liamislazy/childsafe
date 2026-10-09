"""Print dependency versions/import status without disclosing source URLs or credentials."""
from importlib import import_module, metadata

if __name__ == '__main__':
    for package,module in [('torch','torch'),('torchvision','torchvision'),('timm','timm'),
                           ('ultralytics','ultralytics'),('opencv-python','cv2'),
                           ('mediapipe','mediapipe'),('imagehash','imagehash'),('scikit-learn','sklearn')]:
        try:
            import_module(module)
            print(package,metadata.version(package),'OK')
        except Exception as exc:
            print(package,type(exc).__name__,str(exc).splitlines()[0][:200])
