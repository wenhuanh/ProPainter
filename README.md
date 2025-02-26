1. Clone Repo

   ```bash
   git clone https://github.com/wenhuanh/ProPainter.git
   git check out cpu/optimize
   ```

2. Create Conda Environment and Install Dependencies

   ```bash
   # create new anaconda env
   conda create -n propainter python=3.10 -y
   conda activate propainter

   # install python dependencies
   pip3 install --force -r requirements_intel.txt
   ```

3. Adopt CPU optimization with args "--cpu"

   ```bash
   python inference_propainter.py --video inputs/object_removal/bmx-trees --mask inputs/object_removal/bmx-trees_mask --cpu
   ```