ITER=10
meta=("tl")
base=("ridge" "lasso" "dt" "kr")

# Standalone estimators
for MODEL in ${base[@]}
do
    echo $MODEL
    python -W ignore ../main.py --data_path ../datasets/IHDP --dtype ihdp --iter $ITER -o ../results/ihdp_${MODEL} --sr --em $MODEL
done

# Meta models
for MODEL in ${meta[@]}
do
    for BASE_MODEL in ${base[@]}
    do
        echo ${MODEL}_${BASE_MODEL}
        python -W ignore ../main.py --data_path ../datasets/IHDP --dtype ihdp --iter $ITER -o ../results/ihdp_${MODEL}-${BASE_MODEL} --sr --em $MODEL --ebm $BASE_MODEL
    done
done

meta=("sl" "tl")
tuning=("once" "every")
runs=(10 100)
frac=(0.5 0.75 1.0)
for MODEL in ${meta[@]}
do
    for BASE_MODEL in ${base[@]}
    do
        for TUNING_I in ${tuning[@]}
        do
            for RUNS_I in ${runs[@]}
            do
                for FRAC_I in ${frac[@]}
                do
                    echo ${MODEL}_${BASE_MODEL}_${TUNING_I}_${RUNS_I}_${FRAC_I}
                    python -W ignore ../main.py --data_path ../datasets/IHDP --dtype ihdp --iter $ITER -o ../results/ihdp_${MODEL}-${BASE_MODEL}-${TUNING_I}-${RUNS_I}-${FRAC_I} --sr --em $MODEL --ebm $BASE_MODEL --ol --tuning $TUNING_I --n_runs $RUNS_I --t_frac $FRAC_I
                done
            done
        done
    done
done

# Post-processing the results
#python ../results/process.py --data_path ../results --dtype ihdp -o ../results --sm ${simple[@]} --mm ${meta[@]} --bm ${base[@]} --show