mode=trian
method=dl

batch_size=1024
learning_rate=2e-3
epoches=100
beta1=0.99
beta2=0.999
weight_decay=1e-6
eps=1e-8
work_dir=./log/lr4
max_evals=20

eta=0.1
min_child_weight=1
gamma=0
max_depth=6
lambda=0
alpha=0
num_boost_round=100

if [ ${method} = "dl"]
then
    python dlframe/runner/${method}/${mode}.py \
        --batch_size ${batch_size}\
        --learning_rate ${learning_rate}\
        --epoches ${epoches}\
        --beta1 ${beta1}\
        --beta2 ${beta2}\
        --weight_decay ${weight_decay}\
        --eps ${eps}\
        --work_dir ${work_dir}\
        --max_evals ${max_evals}
elif [ ${method} = "ml"]
then
    python dlframe/runner/${method}/${mode}.py \
        --eta ${eta}\
        --min_child_weight ${min_child_weight}\
        --gamma ${gamma}\
        --max_depth ${max_depth}\
        --lambda ${lambda}\
        --alpha ${alpha}\
        --num_boost_round ${num_boost_round}\

fi