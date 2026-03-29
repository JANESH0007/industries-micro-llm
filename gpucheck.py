"""
Hardware Check — run before training
"""
import tensorflow as tf
import sys, platform, time

print("=" * 50)
print("  Hardware Check")
print("=" * 50)
print(f"OS              : {platform.system()} {platform.machine()}")
print(f"Python          : {sys.version.split()[0]}")
print(f"TensorFlow      : {tf.__version__}")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"\n✅ GPU found: {len(gpus)}")
    for g in gpus:
        print(f"   {g.name}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    print("Memory growth  : enabled")
else:
    print("\n⚠  No GPU — will use CPU")
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        print("   Apple Silicon detected. Install tensorflow-metal for GPU.")

print(f"\nCPU count       : {len(tf.config.list_physical_devices('CPU'))}")

print("\nRunning compute test...")
t0 = time.time()
a = tf.random.normal((2000, 2000))
_ = tf.matmul(a, a).numpy()
print(f"Matrix multiply : {time.time()-t0:.3f}s ({'GPU' if gpus else 'CPU'})")
print("\nCheck complete ✔")
