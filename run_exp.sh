for model in "blstm" "brnn"
do
    for feature in "intensity"
    do 
        for dataset in "north_indian_ocean" "south_indian_ocean" "north_west_pacific_ocean" "south_pacific_ocean"
        do
            echo "\n\nTraining $feature $model for $dataset \n"
            python train.py --epochs 120 --model $model --ds-name $dataset --samples 150 --features $feature --lr 0.01
            echo "Done with $feature $model for $dataset"
        done
    done
done
