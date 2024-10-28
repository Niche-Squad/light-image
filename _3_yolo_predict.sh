for i in {1..4}
do
    filename=Beef-0${i}_round2.mp4
    logname=beef-0${i}
    yolo detect predict\
        model=model/yolov8m-brd.pt\
        source=data/videos/${filename}\
        device=mps\
        save_txt=True\
        name=. project=logs/${logname}
done

