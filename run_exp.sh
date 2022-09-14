for model in "brnn" "blstm"
do
    for feature in "location" "intensity"
    do 
        for dataset in "north_indian_ocean" "south_indian_ocean" "north_west_pacific_ocean" "south_pacific_ocean"
        do
            echo "Training $model for $dataset"
            python train.py --epochs 120 --model $model --ds-name $dataset --samples 100 --features $feature
        done
    done
done
