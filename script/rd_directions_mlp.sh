python plot_surface.py --model mlp --dir_file cifar10/trained_nets/mlp/random_mlp_random_direction.h5 \
        --x=-3:3:2000 --dir_type weights --xnorm filter --xignore biasbn --plot \
        --hidden_dims 1000 1000 1000 --batch_size 1000 --loss_name mse --loss_max 10e20