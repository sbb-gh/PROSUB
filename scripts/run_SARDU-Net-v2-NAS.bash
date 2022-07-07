# Dummy bash code to obtain results for SARDU-Net-v2-NAS in table 1
# This runs both the training and evaluation script.
# The final result is the average across the splits i.e. across SPLIT_NO


# Replace with your variables used in main.py:
data_fil=
out_base=

# Activate environment corresponding to PROSUB.yml

# Uncheck one of the two below:
#num_units_init='1063 781 781 1063' # for M=500
num_units_init='417 333 781 1063' # for M<500


T_values='4 8 12 16 20 24 28 32 36'
M_values='1344 500 250 100 50 40 30 20 10'

for SPLIT_NO in {0..4}
do

    RUN_NO=0

    case $SPLIT_NO in
        0) data_train_subjs="0012 0013 0014"; data_val_subjs="0015"; data_test_subjs="0011"; split_name="0011out";;
        1) data_train_subjs="0013 0014 0015"; data_val_subjs="0011"; data_test_subjs="0012"; split_name="0012out";;
        2) data_train_subjs="0014 0015 0011"; data_val_subjs="0012"; data_test_subjs="0013"; split_name="0013out";;
        3) data_train_subjs="0015 0011 0012"; data_val_subjs="0013"; data_test_subjs="0014"; split_name="0014out";;
        4) data_train_subjs="0011 0012 0013"; data_val_subjs="0014"; data_test_subjs="0015"; split_name="0015out";;
    esac

    name=SARDUNet-v2-NAS_run"$RUN_NO"

    python main_nas.py \
    --data_fil $data_fil \
    --data_train_subjs $data_train_subjs \
    --data_val_subjs $data_val_subjs \
    --data_test_subjs $data_test_subjs \
    --proj_name $name \
    --out_base $out_base \
    --seed $RUN_NO \
    --T_values $T_values \
    --M_values $M_values \
    --noepoch 200 \
    --epochs_decay 20 \
    --tuner greedy \
    --num_layers_nas 1 2 3 \
    --network_name sardunet-nas \
    --dropout_nas 0.2 0.0 0.1 0.3 0.4 \
    --data_normalization original \
    --num_units_nas {128..2048..128} \
    --num_units_init $num_units_init
done
