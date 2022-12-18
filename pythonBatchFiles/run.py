import subprocess

if __name__ == "__main__":
    subprocess.run("wt.exe panel serve Interactive_dashboard.ipynb")
    subprocess.run("wt.exe .\Scripts\python.exe .\yolov5\detectFaceMaskandSocialDistancing.py --weights .\models\FaceMask.pt --source 0")