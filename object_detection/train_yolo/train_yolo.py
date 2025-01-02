from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import dump, load
from skopt.callbacks import CheckpointSaver
from skopt import gp_minimize
from ultralytics import YOLO
from skopt.plots import plot_convergence

# Params
epochs = 100
patience = 10
single_class = False
plots = True

print("hello world")

# The list of hyper-parameters we want to optimize. For each one we define the
# bounds, the corresponding scikit-learn parameter name, as well as how to
# sample values from that dimension (`'log-uniform'` for the learning rate)
space  = [
    Real(.8, .99, name='momentum', prior="uniform"),
    Real(0.001, 0.1, name='lr0', prior='log-uniform'),
    Real(0.0001, 0.001, name = "weight_decay", prior='log-uniform')
]

checkpoint_saver = CheckpointSaver("./checkpoint.pkl", compress=9) # keyword arguments will be passed to `skopt.dump`

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set
yolo = YOLO('yolo11n.pt')

# scikit-learn estimator parameters
@use_named_args(space)
def objective(**params):
    yolo.train(data='cocov11.yaml', epochs=epochs, patience=patience, single_cls = single_class, plots = True, **params, val=True)
    valid_results = yolo.val()
    return -valid_results.results_dict['metrics/recall(B)']


res = load('./checkpoint.pkl')
x0 = res.x_iters
y0 = res.func_vals

res_gp = gp_minimize(
    objective, 
    space, 
    n_calls=75, 
    random_state=0, 
    callback=[checkpoint_saver], 
    n_random_starts = 10,
    x0 = x0, 
    y0 = y0,
    )

dump(res_gp, 'result.pkl')

x = plot_convergence(res_gp)
x.figure.savefig("filename.png")

print("""Best parameters:
- momentum=%.6f
- learning_rate=%.6f
- weight_decay=%.6f""" % (res_gp.x[0], res_gp.x[1],
                            res_gp.x[2]))