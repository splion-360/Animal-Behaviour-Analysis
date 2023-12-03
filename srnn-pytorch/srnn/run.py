import subprocess
if __name__ == "__main__":
    print("Executing the train.py file")
    subprocess.call("python srnn/train.py --num_epochs 100 --train True --save_dir ./save_with_gl ")
    #python plot.py --load_dir "../save_no_gl/" --save_dir "12KP_NGL"