source .env
for i in {1..4}
do
    for j in 08 21
    do
        filename=Beef-0${i}_202410${j}_overview.mp4
        echo "Extracting frames from video ${filename}"
        python ${DIR_SRC}/utils/extract_frames.py\
            -i ${DIR_SRC}/data/videos/${filename}\
            -o ${DIR_SRC}/data/images/0${i}-${j}\
            -f 5
    done    
done
