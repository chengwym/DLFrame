mode=train
batch_size=1024
learning_rate=2e-3
epoches=100
beta1=0.99
beta2=0.99
weight_decay=1e-3
eps=1e-5

python dlframe/runner/${mode}.py \
    --batchsize ${batch_size}\
    --learning_rate ${learning_rate}\
    --epoches ${epoches}\
    --beta1 ${beta1}\
    --beta2 ${beta2}\
    --weight_decay ${weight_decay}\
    --eps ${eps}\