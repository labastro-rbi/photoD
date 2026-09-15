"""Plots of the metrics written by training_size.py."""
from matplotlib import pyplot as plt
import pandas


def import_from_csv(path):
    """Read a metrics file written by training_size.py."""
    return pandas.read_csv(path, sep=", ", engine="python")


def plot_error(data, label1="loss", label2="bayes_loss"):
    """Loss of the network and of the Bayesian estimates vs. training set size."""
    fig, ax = plt.subplots()
    ax.plot(data["reduce_size"], data["loss"], label=label1)
    ax.plot(data["reduce_size"], data["bayes_loss"], label=label2)
    ax.set_xlabel("Training size")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig, ax


def plot_chi2(data, prefix=""):
    """Robust width of (estimate - reference) / sigma for Mr, Ar and FeH vs. training set size."""
    fig, ax = plt.subplots()
    ax.plot(data["reduce_size"], data["chi_squaredMr"], label=prefix+"Mr", color="red", linestyle=":")
    ax.plot(data["reduce_size"], data["bayes_chi_squaredMr"], label="Bayes vs True Mr", color="red", linestyle="solid")
    ax.plot(data["reduce_size"], data["chi_squaredAr"], label=prefix+"Ar", color="blue", linestyle=":")
    ax.plot(data["reduce_size"], data["bayes_chi_squaredAr"], label="Bayes vs True Ar", color="blue", linestyle="solid")
    ax.plot(data["reduce_size"], data["chi_squaredFeH"], label=prefix+"FeH", color="green", linestyle=":")
    ax.plot(data["reduce_size"], data["bayes_chi_squaredFeH"], label="Bayes vs True FeH", color="green", linestyle="solid")
    ax.set_xlabel("Training size")
    ax.set_ylabel(r"$\chi^2$")
    ax.legend()
    return fig, ax


def plot_mse(data, prefix):
    """Mean squared error for Mr, Ar and FeH vs. training set size."""
    fig, ax = plt.subplots()
    ax.plot(data["reduce_size"], data["mseMr"], label=prefix+"Mr", color="red", linestyle=":")
    ax.plot(data["reduce_size"], data["bayes_mseMr"], label="Bayes vs True Mr", color="red", linestyle="solid")
    ax.plot(data["reduce_size"], data["mseAr"], label=prefix+"Ar", color="blue", linestyle=":")
    ax.plot(data["reduce_size"], data["bayes_mseAr"], label="Bayes vs True Ar", color="blue", linestyle="solid")
    ax.plot(data["reduce_size"], data["mseFeH"], label=prefix+"FeH", color="green", linestyle=":")
    ax.plot(data["reduce_size"], data["bayes_mseFeH"], label="Bayes vs True FeH", color="green", linestyle="solid")
    ax.set_xlabel("Training size")
    ax.set_ylabel("Mean squared error")
    ax.legend()
    return fig, ax

def plot_training_time(data):
    fig, ax = plt.subplots()
    ax.plot(data["reduce_size"], data["time"])
    ax.set_xlabel("Training size")
    ax.set_ylabel("Training time")
    return fig, ax


d_truth = import_from_csv("../outputs/metrics_truth1.txt")
fig1, ax1 = plot_error(d_truth, label1="NN vs True loss", label2="Bayes vs True loss")
ax1.set_title("Error on Truth")
fig1.savefig("../outputs/error_truth.png")
fig2, ax2 = plot_chi2(d_truth, prefix="NN vs True ")
ax2.set_title(r"$\chi^2$ NN vs Truth")
fig2.savefig("../outputs/chi2_truth.png")
fig3, ax3 = plot_mse(d_truth, prefix="NN vs True ")
print ("MSE NN vs Truth Ar mean:", d_truth["mseAr"].mean(), " STD:", d_truth["mseAr"].std())
print ("MSE Bayes vs True Ar mean", d_truth["bayes_mseAr"].mean(), " STD:", d_truth["bayes_mseAr"].std())
ax3.set_title("MSE NN vs Truth")
fig3.savefig("../outputs/mse_truth.png")
d_bayes = import_from_csv("../outputs/metrics_bayes1.txt")
fig4, ax4 = plot_error(d_bayes, label1="NN vs Bayes loss", label2="Bayes vs True loss")
ax4.set_title("Error on Bayes estimates")
fig4.savefig("../outputs/error_bayes.png")
fig5, ax5 = plot_chi2(d_bayes, prefix="NN vs Bayes ")
ax5.set_title(r"$\chi^2$ NN vs Bayes estimates")
fig5.savefig("../outputs/chi2_bayes.png")
fig6, ax6 = plot_mse(d_bayes, prefix="NN vs Bayes ")
print ("MSE NN vs Bayes Ar mean:", d_bayes["mseAr"].mean(), " STD:", d_bayes["mseAr"].std())
print ("MSE Bayes vs True Ar mean", d_bayes["bayes_mseAr"].mean(), " STD:", d_bayes["bayes_mseAr"].std())
ax6.set_title("MSE NN vs Bayes estimates")
fig6.savefig("../outputs/mse_bayes.png")
fig7, ax7 = plot_training_time(d_truth)
ax7.set_title("Training time")
fig7.savefig("../outputs/training_time.png")
plt.show()
