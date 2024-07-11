python plot_surface.py --model resnet56 --x=-1:1:51 \
--model_file cifar10/trained_nets/resnet56/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --plot