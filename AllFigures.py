import csv
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def fit_(x, y, eps):
    # Init theta
    m, n = x.shape
    theta = np.zeros(n)

    # Newton's method
    while True:
        # Save old theta
        theta_old = np.copy(theta)

        # Compute Hessian Matrix
        h_x = 1 / (1 + np.exp(-x.dot(theta)))
        H = (x.T * h_x * (1 - h_x)).dot(x) / m
        gradient_J_theta = x.T.dot(h_x - y) / m
        # Updata theta
        theta -= np.linalg.inv(H).dot(gradient_J_theta)
        # End training
        if np.linalg.norm(theta - theta_old, ord=1) < eps:
            break

    return theta


def predict(x, theta):
    return 1 / (1 + np.exp(-x.dot(theta)))


# Training set
with open('HockeyTrain.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the header
    CSHX_1 = []
    CSHX_2 = []
    CSH_Y = []
    CSVX_1 = []
    CSVX_2 = []
    CSV_Y = []
    CPP_X_1 = []
    CPP_X_2 = []
    CPP_Y = []
    SHS_X_1 = []
    SHS_X_2 = []
    SHS_Y = []
    SHPP_X_1 = []
    SHPP_X_2 = []
    SHPP_Y = []
    SVPP_X_1 = []
    SVPP_X_2 = []
    SVPP_Y = []

    for row in reader:
        CSHX_1.append(float(row[1]))
        CSHX_2.append(float(row[2]))
        CSH_Y.append(float(row[4]))
        CSVX_1.append(float(row[1]))
        CSVX_2.append(float(row[3]))
        CSV_Y.append(float(row[4]))
        CPP_X_1.append(float(row[1]))
        CPP_X_2.append(float(row[5]))
        CPP_Y.append(float(row[4]))
        SHS_X_1.append(float(row[2]))
        SHS_X_2.append(float(row[3]))
        SHS_Y.append(float(row[4]))
        SHPP_X_1.append(float(row[2]))
        SHPP_X_2.append(float(row[5]))
        SHPP_Y.append(float(row[4]))
        SVPP_X_1.append(float(row[3]))
        SVPP_X_2.append(float(row[5]))
        SVPP_Y.append(float(row[4]))

CSH_X_train = np.array(np.transpose(np.concatenate((np.matrix(CSHX_1), np.matrix(CSHX_2)))))
CSH_Y_train = np.array(((CSH_Y)))
CSV_X_train = np.array(np.transpose(np.concatenate((np.matrix(CSVX_1), np.matrix(CSVX_2)))))
CSV_Y_train = np.array(((CSV_Y)))
CPP_X_train = np.array(np.transpose(np.concatenate((np.matrix(CPP_X_1), np.matrix(CPP_X_2)))))
CPP_Y_train = np.array(((CPP_Y)))
SHS_X_train = np.array(np.transpose(np.concatenate((np.matrix(SHS_X_1), np.matrix(SHS_X_2)))))
SHS_Y_train = np.array(((SHS_Y)))
SHPP_X_train = np.array(np.transpose(np.concatenate((np.matrix(SHPP_X_1), np.matrix(SHPP_X_2)))))
SHPP_Y_train = np.array(((SHPP_Y)))
SVPP_X_train = np.array(np.transpose(np.concatenate((np.matrix(SVPP_X_1), np.matrix(SVPP_X_2)))))
SVPP_Y_train = np.array(((SVPP_Y)))

eps = 1e-5

new_cshx = np.zeros((CSH_X_train.shape[0], CSH_X_train.shape[1] + 1), dtype=CSH_X_train.dtype)
new_cshx[:, 0] = 1
new_cshx[:, 1:] = CSH_X_train
new_csv_x = np.zeros((CSV_X_train.shape[0], CSV_X_train.shape[1] + 1), dtype=CSV_X_train.dtype)
new_csv_x[:, 0] = 1
new_csv_x[:, 1:] = CSV_X_train
new_cpp_x = np.zeros((CPP_X_train.shape[0], CPP_X_train.shape[1] + 1), dtype=CPP_X_train.dtype)
new_cpp_x[:, 0] = 1
new_cpp_x[:, 1:] = CPP_X_train
new_shs_x = np.zeros((SHS_X_train.shape[0], SHS_X_train.shape[1] + 1), dtype=SHS_X_train.dtype)
new_shs_x[:, 0] = 1
new_shs_x[:, 1:] = SHS_X_train
new_shpp_x = np.zeros((SHPP_X_train.shape[0], SHPP_X_train.shape[1] + 1), dtype=SHPP_X_train.dtype)
new_shpp_x[:, 0] = 1
new_shpp_x[:, 1:] = SHPP_X_train
svpp_new_x = np.zeros((SVPP_X_train.shape[0], SVPP_X_train.shape[1] + 1), dtype=SVPP_X_train.dtype)
svpp_new_x[:, 0] = 1
svpp_new_x[:, 1:] = SVPP_X_train

cshx_theta = fit_(new_cshx, CSH_Y_train, eps)
csv_theta = fit_(new_csv_x, CSV_Y_train, eps)
cpp_theta = fit_(new_cpp_x, CPP_Y_train, eps)
shs_theta = fit_(new_shs_x, SHS_Y_train, eps)
shpp_theta = fit_(new_shpp_x, SHPP_Y_train, eps)
svpp_theta = fit_(svpp_new_x, SVPP_Y_train, eps)

# Plot decision boundary (found by solving for theta^T x = 0)
csh_margin = (max(new_cshx[:, -2]) - min(new_cshx[:, -2])) * 0.2
csv_margin = (max(new_csv_x[:, -2]) - min(new_csv_x[:, -2])) * 0.2
cpp_margin = (max(new_cpp_x[:, -2]) - min(new_cpp_x[:, -2])) * 0.2
shs_margin = (max(new_shs_x[:, -2]) - min(new_shs_x[:, -2])) * 0.2
shpp_margin = (max(new_shpp_x[:, -2]) - min(new_shpp_x[:, -2])) * 0.2
svpp_margin = (max(svpp_new_x[:, -2]) - min(svpp_new_x[:, -2])) * 0.2

plt.figure()
plt.plot(new_cshx[CSH_Y_train == 1, -2], new_cshx[CSH_Y_train == 1, -1], 'cx', linewidth=2)
plt.plot(new_cshx[CSH_Y_train == 0, -2], new_cshx[CSH_Y_train == 0, -1], 'mx', linewidth=2)
csh_x1 = np.arange(min(new_cshx[:, -2]) - csh_margin, max(new_cshx[:, -2]) + csh_margin, 0.01)
csh_x2 = -(cshx_theta[0] / cshx_theta[2] + cshx_theta[1] / cshx_theta[2] * csh_x1)
plt.plot(csh_x1, csh_x2, c='black', linewidth=2)
plt.ylabel('Shooting Percentage')
plt.xlabel('Corsi For Percentage')
plt.figure()
plt.plot(new_csv_x[CSV_Y_train == 1, -2], new_csv_x[CSV_Y_train == 1, -1], 'cx', linewidth=2)
plt.plot(new_csv_x[CSV_Y_train == 0, -2], new_csv_x[CSV_Y_train == 0, -1], 'mo', linewidth=2)
csv_x1 = np.arange(min(new_csv_x[:, -2]) - csv_margin, max(new_csv_x[:, -2]) + csv_margin, 0.01)
csv_x2 = -(csv_theta[0] / csv_theta[2] + csv_theta[1] / csv_theta[2] * csv_x1)
plt.plot(csv_x1, csv_x2, c='red', linewidth=2)
plt.ylabel('Save Percentage')
plt.xlabel('Corsi For Percentage')
plt.figure()
plt.plot(new_cpp_x[CPP_Y_train == 1, -2], new_cpp_x[CPP_Y_train == 1, -1], 'cx', linewidth=2)
plt.plot(new_cpp_x[CPP_Y_train == 0, -2], new_cpp_x[CPP_Y_train == 0, -1], 'mx', linewidth=2)
cpp_x1 = np.arange(min(new_cpp_x[:, -2]) - cpp_margin, max(new_cpp_x[:, -2]) + cpp_margin, 0.01)
cpp_x2 = -(cpp_theta[0] / cpp_theta[2] + cpp_theta[1] / cpp_theta[2] * cpp_x1)
plt.plot(cpp_x1, cpp_x2, c='black', linewidth=2)
plt.ylabel('Power Play Percentage')
plt.xlabel('Corsi For Percentage')
plt.figure()
plt.plot(new_shs_x[SHS_Y_train == 1, -2], new_shs_x[SHS_Y_train == 1, -1], 'cx', linewidth=2)
plt.plot(new_shs_x[SHS_Y_train == 0, -2], new_shs_x[SHS_Y_train == 0, -1], 'mo', linewidth=2)
shs_x1 = np.arange(min(new_shs_x[:, -2]) - shs_margin, max(new_shs_x[:, -2]) + shs_margin, 0.01)
shs_x2 = -(shs_theta[0] / shs_theta[2] + shs_theta[1] / shs_theta[2] * shs_x1)
plt.plot(shs_x1, shs_x2, c='red', linewidth=2)
plt.ylabel('Save Percentage')
plt.xlabel('Shooting Percentage')
plt.figure()
plt.plot(new_shpp_x[SHPP_Y_train == 1, -2], new_shpp_x[SHPP_Y_train == 1, -1], 'cx', linewidth=2)
plt.plot(new_shpp_x[SHPP_Y_train == 0, -2], new_shpp_x[SHPP_Y_train == 0, -1], 'mx', linewidth=2)
shpp_x1 = np.arange(min(new_shpp_x[:, -2]) - shpp_margin, max(new_shpp_x[:, -2]) + shpp_margin, 0.01)
shpp_x2 = -(shpp_theta[0] / shpp_theta[2] + shpp_theta[1] / shpp_theta[2] * shpp_x1)
plt.plot(shpp_x1, shpp_x2, c='black', linewidth=2)
plt.ylabel('Power Play Percentage')
plt.xlabel('Shooting Percentage')
plt.figure()
plt.plot(svpp_new_x[SVPP_Y_train == 1, -2], svpp_new_x[SVPP_Y_train == 1, -1], 'cx', linewidth=2)
plt.plot(svpp_new_x[SVPP_Y_train == 0, -2], svpp_new_x[SVPP_Y_train == 0, -1], 'mo', linewidth=2)
svpp_x1 = np.arange(min(svpp_new_x[:, -2]) - svpp_margin, max(svpp_new_x[:, -2]) + svpp_margin, 0.01)
svpp_x2 = -(svpp_theta[0] / svpp_theta[2] + svpp_theta[1] / svpp_theta[2] * svpp_x1)
plt.plot(svpp_x1, svpp_x2, c='red', linewidth=2)
plt.ylabel('Power Play Percentage')
plt.xlabel('Save Percentage')

# Validation code
with open('HockeyValid.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the header
    CSH_XV_1 = []
    CSH_XV_2 = []
    CSH_YV = []
    CSV_XV_1 = []
    CSV_XV_2 = []
    CSV_YV = []
    CPP_XV_1 = []
    CPP_XV_2 = []
    CPP_YV = []
    SHS_XV_1 = []
    SHS_XV_2 = []
    SHS_YV = []
    SHPP_XV_1 = []
    SHPP_XV_2 = []
    SHPP_YV = []
    SVPP_XV_1 = []
    SVPP_XV_2 = []
    SVPP_YV = []

    for row in reader:
        CSH_XV_1.append(float(row[1]))
        CSH_XV_2.append(float(row[2]))
        CSH_YV.append(float(row[4]))
        CSV_XV_1.append(float(row[1]))
        CSV_XV_2.append(float(row[3]))
        CSV_YV.append(float(row[4]))
        CPP_XV_1.append(float(row[1]))
        CPP_XV_2.append(float(row[5]))
        CPP_YV.append(float(row[4]))
        SHS_XV_1.append(float(row[2]))
        SHS_XV_2.append(float(row[3]))
        SHS_YV.append(float(row[4]))
        SHPP_XV_1.append(float(row[2]))
        SHPP_XV_2.append(float(row[5]))
        SHPP_YV.append(float(row[4]))
        SVPP_XV_1.append(float(row[3]))
        SVPP_XV_2.append(float(row[5]))
        SVPP_YV.append(float(row[4]))

CSH_X_Valid = np.array(np.transpose(np.concatenate((np.matrix(CSH_XV_1), np.matrix(CSH_XV_2)))))
CSV_X_Valid = np.array(np.transpose(np.concatenate((np.matrix(CSV_XV_1), np.matrix(CSV_XV_2)))))
CPP_X_Valid = np.array(np.transpose(np.concatenate((np.matrix(CPP_XV_1), np.matrix(CPP_XV_2)))))
SHS_X_Valid = np.array(np.transpose(np.concatenate((np.matrix(SHS_XV_1), np.matrix(SHS_XV_2)))))
SHPP_X_Valid = np.array(np.transpose(np.concatenate((np.matrix(SHPP_XV_1), np.matrix(SHPP_XV_2)))))
SVPP_X_Valid = np.array(np.transpose(np.concatenate((np.matrix(SVPP_XV_1), np.matrix(SVPP_XV_2)))))

new_csh_x_Valid = np.zeros((CSH_X_Valid.shape[0], CSH_X_Valid.shape[1] + 1), dtype=CSH_X_Valid.dtype)
new_csh_x_Valid[:, 0] = 1
new_csh_x_Valid[:, 1:] = CSH_X_Valid
new_csvx_Valid = np.zeros((CSV_X_Valid.shape[0], CSV_X_Valid.shape[1] + 1), dtype=CSV_X_Valid.dtype)
new_csvx_Valid[:, 0] = 1
new_csvx_Valid[:, 1:] = CSV_X_Valid
new_cpp_x_Valid = np.zeros((CPP_X_Valid.shape[0], CPP_X_Valid.shape[1] + 1), dtype=CPP_X_Valid.dtype)
new_cpp_x_Valid[:, 0] = 1
new_cpp_x_Valid[:, 1:] = CPP_X_Valid
new_shs_x_Valid = np.zeros((SHS_X_Valid.shape[0], SHS_X_Valid.shape[1] + 1), dtype=SHS_X_Valid.dtype)
new_shs_x_Valid[:, 0] = 1
new_shs_x_Valid[:, 1:] = SHS_X_Valid
new_shpp_x_Valid = np.zeros((SHPP_X_Valid.shape[0], SHPP_X_Valid.shape[1] + 1), dtype=SHPP_X_Valid.dtype)
new_shpp_x_Valid[:, 0] = 1
new_shpp_x_Valid[:, 1:] = SHPP_X_Valid
new_svpp_x_Valid = np.zeros((SVPP_X_Valid.shape[0], SVPP_X_Valid.shape[1] + 1), dtype=SVPP_X_Valid.dtype)
new_svpp_x_Valid[:, 0] = 1
new_svpp_x_Valid[:, 1:] = SVPP_X_Valid

CSH_Y_Valid = np.array(((CSH_YV)))
CSV_Y_Valid = np.array(((CSV_YV)))
CPP_Y_Valid = np.array(((CPP_YV)))
SHS_Y_Valid = np.array(((SHS_YV)))
SHPP_Y_Valid = np.array(((SHPP_YV)))
SVPP_Y_Valid = np.array(((SVPP_YV)))

CSH_Y_Predict = predict(new_csh_x_Valid, cshx_theta)
CSV_Y_Predict = predict(new_csvx_Valid, csv_theta)
CPP_Y_Predict = predict(new_cpp_x_Valid, cpp_theta)
SHS_Y_Predict = predict(new_shs_x_Valid, shs_theta)
SHPP_Y_Predict = predict(new_shpp_x_Valid, shpp_theta)
SVPP_Y_Predict = predict(new_svpp_x_Valid, svpp_theta)

CSH_Y_Pred_Out = (CSH_Y_Predict > 0.5)
CSV_Y_Pred_Out = (CSV_Y_Predict > 0.5)
CPP_Y_Pred_Out = (CPP_Y_Predict > 0.5)
SHS_Y_Pred_Out = (SHS_Y_Predict > 0.5)
SHPP_Y_Pred_Out = (SHPP_Y_Predict > 0.5)
SVPP_Y_Pred_Out = (SVPP_Y_Predict > 0.5)

csh_sum = np.sum(CSH_Y_Valid == CSH_Y_Pred_Out)
csh_total = len(CSH_Y_Pred_Out)
print("Prediction with Corsi and Shooting Percentage is: " + str((csh_sum / csh_total) * 100) + "% accurate")
csv_sum = np.sum(CSV_Y_Valid == CSV_Y_Pred_Out)
csv_total = len(CSV_Y_Pred_Out)
print("Prediction with Corsi and Save Percentage is: " + str((csv_sum / csv_total) * 100) + "% accurate")
cpp_sum = np.sum(CPP_Y_Valid == CPP_Y_Pred_Out)
cpp_total = len(CPP_Y_Pred_Out)
print("Prediction with Corsi and Power Play Percentage is: " + str((cpp_sum / cpp_total) * 100) + "% accurate")
shs_sum = np.sum(SHS_Y_Valid == SHS_Y_Pred_Out)
shs_total = len(SHS_Y_Pred_Out)
print("Prediction with Shooting and Save Percentage is: " + str((shs_sum / shs_total) * 100) + "% accurate")
shpp_sum = np.sum(SHPP_Y_Valid == SHPP_Y_Pred_Out)
shpp_total = len(SHPP_Y_Pred_Out)
print("Prediction with Shooting and Power Play Percentage is: " + str((shpp_sum / shpp_total) * 100) + "% accurate")
svpp_sum = np.sum(SVPP_Y_Valid == SVPP_Y_Pred_Out)
svpp_total = len(SVPP_Y_Pred_Out)
print("Prediction with Save and Power Play Percentage is: " + str((svpp_sum / svpp_total) * 100) + "% accurate")

plt.show()