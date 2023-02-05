mode=train
batch_size=1024
learning_rate=2e-3
epoches=100
beta1=0.99
beta2=0.999
weight_decay=1e-6
eps=1e-8
work_dir=./log/lr4
max_evals=20

python dlframe/runner/${mode}.py \
    --batch_size ${batch_size}\
    --learning_rate ${learning_rate}\
    --epoches ${epoches}\
    --beta1 ${beta1}\
    --beta2 ${beta2}\
    --weight_decay ${weight_decay}\
    --eps ${eps}\
    --work_dir ${work_dir}\
    --max_evals ${max_evals}