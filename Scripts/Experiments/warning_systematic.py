import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

from BasicTools.experiment_info import ExperimentGenerator
from Policies.scp_mpc import SCPsolve, LinearOLsolve
from WarningSystem.verifying_warning import generate_warning_rollouts, estimate_warning_stats, plot_warning_error, plot_warning_omegas, plot_roc, plot_quants
from WarningSystem.CP_alerter import CPAlertSystem
from WarningSystem.WarningBaselines.NearestSafe_baseline import NearestSafeAlertSystem
from WarningSystem.WarningBaselines.NN_baseline import NNAlertSystem
from WarningSystem.WarningBaselines.RF_baseline import RFAlertSystem
from WarningSystem.WarningBaselines.SVM_baseline import SVMAlertSystem, OutlierSVMAlertSystem

if __name__ == '__main__':
    #### User Settings ####
    EXP_NAME = 'pos' # 'pos', 'speed', 'cbf'
    SYS_NAME = 'body' # 'body', 'linear'
    EXP_DIR = os.path.join('..', 'data', EXP_NAME + '_' + SYS_NAME)
    
    POLICY_NAME = 'mpc'

    output_dir = '../Figures/warning_systematic_figs'

    # Whether to load (or generate) the training and test rollouts
    LOAD_DATA = True
    # Whether to save the training and test rollouts
    SAVE_DATA = False
    # Whether to load (or generate) the results of NNCP ablations
    LOAD_NNCP = True
    # Whether to load (or generate) the results of ML comparisons
    LOAD_ML = True
    # Whether to save the albation and ML results and figures
    SAVE = False

    # Fix the random seed
    np.random.seed(0)

    verbose = True

    num_test = 500
    num_fit = 25
    num_reps = 20
    num_eps = 10
    num_cp = 10

    #### Load the experiment generator #### 
    exp_gen = pickle.load(open(os.path.join(EXP_DIR, 'exp_gen.pkl'), 'rb'))
    bounds = np.array([[-12.5,12.5],[-12.5,12.5],[0,7.5]])

    #### Load the policy ####
    policy = pickle.load(open(os.path.join(EXP_DIR, POLICY_NAME + '_policy.pkl'), 'rb'))

    #### List the Baselines ####

    # Our method
    # Previously had subselect = None
    lrt_cp_alerter = CPAlertSystem(transformer=None, pwr=True, type_flag='lrt', subselect=None)

    # NNCP Ablations: instead of two-sample, either just distance to nearest unsafe or just distance to nearest safe
    norm_cp_alerter = CPAlertSystem(transformer=None, pwr=True, type_flag='norm')
    nearest_safe_alerter = NearestSafeAlertSystem(transformer=None, p=2, workers=-1)

    # ML Baselines with hold-out CP
    rf_alerter = RFAlertSystem(balance=True, verbose=False, num_cp=num_cp, **{'n_estimators':500, 'max_features':'sqrt', 'max_depth':8})
    no_bal_rf_alerter = RFAlertSystem(balance=False, verbose=False, num_cp=num_cp, **{'n_estimators':500, 'max_features':'sqrt', 'max_depth':8})

    nn_alerter = NNAlertSystem(layer_sizes=[9,100,50,25,1],optimizer_args={'lr':1e-3, 'betas':(0.9, 0.999), 'weight_decay': 0},
                                batch_size=256,epochs=50,balance=True,num_cp=num_cp,verbose=False)
    no_bal_nn_alerter = NNAlertSystem(layer_sizes=[9,100,50,25,1],optimizer_args={'lr':1e-3, 'betas':(0.9, 0.999), 'weight_decay': 0},
                                batch_size=256,epochs=50,balance=False,num_cp=num_cp,verbose=False)
    
    svm_alerter = SVMAlertSystem(balance=True, verbose=False, num_cp=num_cp, svm_type='SVC', **{'C':1e4, 'kernel':'rbf'})
    no_bal_svm_alerter = SVMAlertSystem(balance=False, verbose=False, num_cp=num_cp, svm_type='SVC', **{'C':1e4, 'kernel':'rbf'})

    # Ablation which reuses training points for CP (invalidating it)
    # reuse_rf_alerter = RFAlertSystem(balance=False, verbose=False, num_cp=0, **{'n_estimators':500, 'max_features':'sqrt', 'max_depth':8})
    # outlier_svm_alerter = OutlierSVMAlertSystem(verbose=False, **{'nu':0.5})

    nncp_alert_systems = [lrt_cp_alerter, norm_cp_alerter, nearest_safe_alerter]
    nncp_names = ['Unsafe-Safe', 'Unsafe', '-Safe'] 
    nncp_colors = ['blue', 'red', 'green']

    ml_alert_systems = [nn_alerter, no_bal_nn_alerter, rf_alerter, no_bal_rf_alerter, svm_alerter, no_bal_svm_alerter]
    ml_names = ['Bal-NN', 'NN', 'Bal-RF', 'RF', 'Bal-SVM', 'SVM']
    ml_colors = ['darkred', 'red', 'darkgreen', 'lime', 'darkmagenta', 'magenta']

    # Classifier systems: no calibration, use default cutoff
    nn_classifier = NNAlertSystem(layer_sizes=[9,100,50,25,1],optimizer_args={'lr':1e-3, 'betas':(0.9, 0.999), 'weight_decay': 0},
                                batch_size=256,epochs=50,balance=True,num_cp=0,verbose=False)
    no_bal_nn_classifier = NNAlertSystem(layer_sizes=[9,100,50,25,1],optimizer_args={'lr':1e-3, 'betas':(0.9, 0.999), 'weight_decay': 0},
                                batch_size=256,epochs=50,balance=True,num_cp=0,verbose=False)

    rf_classifier = RFAlertSystem(balance=True, verbose=False, num_cp=0, **{'n_estimators':500, 'max_features':'sqrt', 'max_depth':8})
    no_bal_rf_classifier = RFAlertSystem(balance=False, verbose=False, num_cp=0, **{'n_estimators':500, 'max_features':'sqrt', 'max_depth':8})

    svm_classifier = SVMAlertSystem(balance=True, verbose=False, num_cp=0, svm_type='SVC', **{'C':1e4, 'kernel':'rbf'})
    no_bal_svm_classifier = SVMAlertSystem(balance=False, verbose=False, num_cp=0, svm_type='SVC', **{'C':1e4, 'kernel':'rbf'})

    classifier_systems = [nn_classifier, no_bal_nn_classifier, rf_classifier, no_bal_rf_classifier, svm_classifier, no_bal_svm_classifier]
    classifier_names = ['Bal NN', 'NN', 'Bal RF', 'RF', 'Bal SVM', 'SVM']
    classifier_colors = ['darkred', 'red', 'darkgreen', 'lime', 'darkmagenta', 'magenta']

    #### Generate the Data ####

    if LOAD_DATA:
        test_rollouts = pickle.load(open(os.path.join(EXP_DIR, 'test_rollouts'), 'rb'))
        train_rollouts_list = pickle.load(open(os.path.join(EXP_DIR, 'train_rollouts_list'), 'rb'))
        beta = test_rollouts.count_subset('crash') / test_rollouts.num_runs
    else:
        test_rollouts, train_rollouts_list, beta = generate_warning_rollouts(num_reps, num_test, num_fit, exp_gen, policy, verbose)
        if SAVE_DATA:
            pickle.dump(test_rollouts, open(os.path.join(EXP_DIR, 'test_rollouts'), 'wb'))
            pickle.dump(train_rollouts_list, open(os.path.join(EXP_DIR, 'train_rollouts_list'), 'wb'))

    # Copy test_rollouts over to form a list
    test_rollouts_list = [test_rollouts] * len(train_rollouts_list)

    # Make sure to choose something compatible with cp_rf_alerter i.e.
    # can only go from 1/(num_cp + 1) to num_cp/(num_cp+1)
    # Add +1e-5 for numerical reasons, ensure don't go below 1/(num_fit+1)
    eps_vals = np.linspace(1/(num_cp+1), num_cp/(num_cp+1), num_eps) + 1e-5
    our_eps_vals = np.linspace(1/(num_fit+1), num_fit/(num_fit+1), num_fit) + 1e-5

    #### Run NNCP Ablations ####
    
    if LOAD_NNCP:
        nncp_conditional_probs_arr = pickle.load(open(os.path.join(EXP_DIR, 'nncp_conditional_probs_arr'), 'rb'))
        nncp_total_probs_arr = pickle.load(open(os.path.join(EXP_DIR, 'nncp_total_probs_arr'), 'rb'))
        nncp_omega_probs_arr = pickle.load(open(os.path.join(EXP_DIR, 'nncp_omega_probs_arr'), 'rb'))
    else:
        nncp_conditional_probs_arr, nncp_total_probs_arr, nncp_omega_probs_arr = \
            estimate_warning_stats(test_rollouts_list, train_rollouts_list, nncp_alert_systems, our_eps_vals, verbose)
        if SAVE:
            pickle.dump(nncp_conditional_probs_arr, open(os.path.join(EXP_DIR, 'nncp_conditional_probs_arr'), 'wb'))
            pickle.dump(nncp_total_probs_arr, open(os.path.join(EXP_DIR, 'nncp_total_probs_arr'), 'wb'))
            pickle.dump(nncp_omega_probs_arr, open(os.path.join(EXP_DIR, 'nncp_omega_probs_arr'), 'wb'))

    # Check whether miss rate bound holds
    ax1, ax2, avg_conditional_probs, quant_conditional_probs, avg_total_probs, quant_total_probs = \
        plot_warning_error(our_eps_vals, nncp_conditional_probs_arr, nncp_total_probs_arr, beta, nncp_names, None, nncp_colors)

    # Plot power curve
    ax3, nncp_avg_omegas, nncp_quant_omegas = plot_warning_omegas(our_eps_vals, nncp_omega_probs_arr, nncp_names, None, nncp_colors)

    #### Run ML Baselines ####

    if LOAD_ML:
        ml_conditional_probs_arr = pickle.load(open(os.path.join(EXP_DIR, 'ml_conditional_probs_arr'), 'rb'))
        ml_total_probs_arr = pickle.load(open(os.path.join(EXP_DIR, 'ml_total_probs_arr'), 'rb'))
        ml_omega_probs_arr = pickle.load(open(os.path.join(EXP_DIR, 'ml_omega_probs_arr'), 'rb'))

        clas_conditional_probs_arr = pickle.load(open(os.path.join(EXP_DIR, 'clas_conditional_probs_arr'), 'rb'))
        clas_total_probs_arr = pickle.load(open(os.path.join(EXP_DIR, 'clas_total_probs_arr'), 'rb'))
        clas_omega_probs_arr = pickle.load(open(os.path.join(EXP_DIR, 'clas_omega_probs_arr'), 'rb'))
    else:
        # ML alert baselines
        ml_conditional_probs_arr, ml_total_probs_arr, ml_omega_probs_arr = \
            estimate_warning_stats(test_rollouts_list, train_rollouts_list, ml_alert_systems, eps_vals, verbose)

        # Classifier baselines, use eps=-1 only
        clas_conditional_probs_arr, clas_total_probs_arr, clas_omega_probs_arr = \
            estimate_warning_stats(test_rollouts_list, train_rollouts_list, classifier_systems, [-1], verbose)

        if SAVE:
            pickle.dump(ml_conditional_probs_arr, open(os.path.join(EXP_DIR, 'ml_conditional_probs_arr'), 'wb'))
            pickle.dump(ml_total_probs_arr, open(os.path.join(EXP_DIR, 'ml_total_probs_arr'), 'wb'))
            pickle.dump(ml_omega_probs_arr, open(os.path.join(EXP_DIR, 'ml_omega_probs_arr'), 'wb'))

            pickle.dump(clas_conditional_probs_arr, open(os.path.join(EXP_DIR, 'clas_conditional_probs_arr'), 'wb'))
            pickle.dump(clas_total_probs_arr, open(os.path.join(EXP_DIR, 'clas_total_probs_arr'), 'wb'))
            pickle.dump(clas_omega_probs_arr, open(os.path.join(EXP_DIR, 'clas_omega_probs_arr'), 'wb'))

    # Plot power curve
    ax4, ml_avg_omegas, ml_quant_omegas = plot_warning_omegas(eps_vals, ml_omega_probs_arr, ml_names, None, ml_colors)
    
    # Add our method to plot
    plot_quants(our_eps_vals, [nncp_omega_probs_arr[0,:,:]], ['Ours'], quants=None, ax=ax4, colors=['blue'])

    # Overlay the classifiers
    for i, name in enumerate(classifier_names):
        # Start: num_alert, num_reps, num_eps -> select i: num_reps, num_eps -> mean, axis=1: num_eps (and num_eps=1)
        eps_act = np.mean(clas_conditional_probs_arr[i], axis=0)
        omega = np.mean(clas_omega_probs_arr[i], axis=0)
        # In fact, just a single point
        if classifier_colors is not None:
            color = classifier_colors[i]
        else:
            color = None

        # Do not label the classifiers since shown in same color
        ax4.scatter(eps_act, omega, label='', marker='x', s=40, color=color)

    # If too cluttered can place outside plot    
    # ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax4.legend(loc='lower right')

    #### Save figures ####

    if SAVE:
        names = ['conditional_miss', 'total_miss', 'omega', 'ml_baselines']
        axes = [ax1, ax2, ax3, ax4]
        figsizes = [(5,3),(5,3),(5,3),(5,3)]

        for i, name in enumerate(names):
            ax = axes[i]
            figsize = figsizes[i]
            full_name = os.path.join(output_dir, name)
            fig = ax.get_figure()

            # Resize
            fig.set_size_inches(*figsize) 
            fig.tight_layout()

            fig.savefig(full_name + '.svg')
            fig.savefig(full_name + '.png')

    plt.show()