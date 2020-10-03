# Suturing for dVRK
## Setup guide
1. Follow [all of the installation instructions here,](https://github.com/divyabudihal/autonomous_surgical_camera) up to the "Autonomous Surgical Camera" heading
2. Navigate to the dvrk-ros respository and comment out line 67 of psm.py, to mute a bunch of extremely annoying warnings
3. Install jupyter notebook with `pip install jupyter` and/or `pip install notebook` - I forgot how I actually did this 
4. Install nbconvert by [following this guide](https://jupytext.readthedocs.io/en/latest/install.html) - I forgot how I did this too but I think this guide works
5. Launch the dvrk with the dvrk_full_cart_simulated.launch launchfile in this repository.
6. The most up-to-date simulation scene is `dVRK-suturing_center_wound.ttt`