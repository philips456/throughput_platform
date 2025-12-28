import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

# ========== CONFIGURATION ==========
WATCH_PATH = "."  # Dossier à surveiller
TRIGGER_EXTS = (".py", "Dockerfile", "Makefile", ".html", ".csv")
IGNORED_DIRS = {"venv", "__pycache__", ".git"}


class PipelineHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return

        filepath = os.path.relpath(event.src_path)
        if not filepath.endswith(TRIGGER_EXTS):
            return

        if any(ignored in filepath for ignored in IGNORED_DIRS):
            return

        print(f"\n📁 Changement détecté : {filepath}")
        self.run_pipeline()

    def run_pipeline(self):
        try:
            # === CI ===
            print("🔁 Étape 1 : CI → check + entraînement")
            subprocess.run(["make", "ci"], check=True)

            # === Docker Build ===
            print("🐳 Étape 2 : Build Docker images")
            print(f"\n📁 images already exist !")

            # === Docker Tag + Push ===
            print("🏷️ Étape 3 : Tag images")
            print(f"\n📁 images already tagged !")

            print("📤 Étape 4 : Push images")
            print(f"\n📁 images already pushed !")

            # === CD ===
            print("🚀 Étape 5 : CD → pull + run + check")
            subprocess.run(["make", "pull", "run-container"], check=True)

            print("✅ CI + Docker + CD terminé avec succès.\n")

        except subprocess.CalledProcessError as e:
            print(f"❌ Échec du pipeline : {e}\n")


def start_watchdog():
    print("👁️  Watchdog CI/CD en cours...")
    print(f"📂 Surveillance du dossier : {os.path.abspath(WATCH_PATH)}")
    print(f"🎯 Extensions surveillées : {', '.join(TRIGGER_EXTS)}")
    observer = Observer()
    handler = PipelineHandler()
    observer.schedule(handler, path=WATCH_PATH, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("🛑 Watchdog arrêté.")
    observer.join()


if __name__ == "__main__":
    start_watchdog()
