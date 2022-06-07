import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("chess_games.csv", delimiter=",")

# COLORS (customized) -> HEX format
BLACK = "#202020"
BLUE = "#1f77b4"
GRAY = "#606060"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
RED = "#d62728"
WHITE = "#f0f0f0"

# FONT SIZES
conf_title = 20
title_size = 16
label_size = 10

# CREATING COLUMNS
# "chess_time" Column
match_time = df["last_move_at"] - df["created_at"]
df["chess_time"] = match_time.apply(lambda time: round(time / (1000 * 60), 2))    # UNIX time conversion to minutes
# "rating_difference" Column
df["rating_difference"] = abs((df["white_rating"] - df["black_rating"]))

# DATA VISUALIZATON
# Pie Chart
pie_exp = 0.05
    # Win rate
wr = df["winner"].value_counts(normalize = True)
wr_labels = ["White","Black","Draw"]
wr_colors = [WHITE, BLACK, GRAY]

fig = plt.figure(figsize = (6,6))
patches, texts, autotexts = plt.pie(wr, labels = wr_labels, colors = wr_colors, explode = [pie_exp, pie_exp, pie_exp], shadow = True, autopct = "%.1f%%", startangle = 120)
autotexts[1].set_color(WHITE)
autotexts[2].set_color(WHITE)
plt.title("Win Rates of Chess Sides (White vs. Black)", fontsize = title_size)
plt.show()
    # Rated/Unrated
rated = df["rated"].value_counts(normalize = True)
rated_labels = ["Rated","Unrated"]
rated_colors = [BLUE, ORANGE]

fig = plt.figure(figsize = (6,6))
plt.pie(rated, labels = rated_labels, colors = rated_colors, explode = [pie_exp, pie_exp], shadow = True, autopct = "%.1f%%")
plt.title("Game Mode Percentage", fontsize = title_size)
plt.show()
    # Victory status
vic_status = df["victory_status"].value_counts(normalize = True)
vic_labels = ["Resign","Checkmate","Out of time","Draw"]
vic_colors = [ORANGE, GREEN, BLUE, RED]

fig = plt.figure(figsize = (6,6))
plt.pie(vic_status, labels = vic_labels, colors = vic_colors, explode = [pie_exp, pie_exp, pie_exp, pie_exp], shadow = True, autopct = "%.1f%%", startangle = 90)
plt.title("End Status of Chess Games", fontsize = title_size)
plt.show()

# Histogram
bin_numbers = 20
    # Rating
player_rating = pd.concat([df["white_rating"], df["black_rating"]])  # all players

fig = plt.figure(figsize = (9,7))
sb.histplot(data = player_rating, bins = bin_numbers, element = "step", kde = True, color = BLACK)
plt.title("Disturbution of Chess Ratings", fontsize = title_size)
plt.xlabel("Player ratings")
plt.xticks(ticks = np.linspace(750, 2750, 21), rotation = 60)
plt.ylabel("Number of players")
plt.show()

# Bar Chart
bar_width = 0.4
    # Most preferred openings (ECO codes)
ecogeneral = df["opening_eco"].value_counts().head()
eco2000 = df.query("white_rating > 2000 & black_rating > 2000")["opening_eco"].value_counts().head()
eco1400 = df.query("white_rating >= 1400 & black_rating >= 1400 & white_rating <= 2000 & black_rating <= 2000")["opening_eco"].value_counts().head()
eco750 = df.query("white_rating < 1400 & black_rating < 1400")["opening_eco"].value_counts().head()

fig,axs = plt.subplots(2, 2, figsize = (9,9))
fig.suptitle("Top 5 Most Preferred Chess Openings", fontsize = title_size)
fig.supxlabel("ECO codes", fontsize = label_size)
fig.supylabel("Count", fontsize = label_size)
axs[0,0].bar(ecogeneral.index.tolist(), ecogeneral, width = bar_width, color = BLUE)
axs[0,0].set_title("General")
axs[0,1].bar(eco750.index.tolist(), eco750, width = bar_width, color = RED)
axs[0,1].set_title("Under 1400 rating")
axs[1,0].bar(eco1400.index.tolist(), eco1400, width = bar_width, color = GREEN)
axs[1,0].set_title("1400-2000 rating")
axs[1,1].bar(eco2000.index.tolist(), eco2000, width = bar_width, color = ORANGE)
axs[1,1].set_title("Over 2000 rating")
plt.tight_layout()
plt.show()

# Box plot
    # Winner and turns
fig = plt.figure(figsize = (9,7))
sb.boxplot(data = df, x = "victory_status", y = "turns")
plt.title("End Game Status & Turns", fontsize = title_size)
plt.xlabel("End game status")
plt.ylabel("Number of turns")
plt.show()

# Strip plot
    # Rated and rating difference    
fig = plt.figure(figsize = (9,7))
sb.stripplot(data = df, x = "rated", y = "rating_difference", hue = "victory_status")
plt.title("Game Type & Rating Difference", fontsize = title_size)
plt.xlabel("Rated")
plt.ylabel("Rating difference")
plt.show()

# Scatter Plot
line_width = 0.2
size1 = 20
size2 = 40
    # Chess time and turns
fig = plt.figure(figsize = (10,8))
sb.scatterplot(data = df, x = "turns", y = "chess_time", color = RED, s = size2, linewidth = line_width, edgecolor = WHITE)
plt.title("Turns & Chess Time", fontsize = title_size)
plt.xlabel("Number of turns")
plt.ylabel("Minutes")
plt.show()
    # Under 250 minutes chess time and turns
under240 = df[df["chess_time"] < 240]
    
fig = plt.figure(figsize = (10,8))
sb.scatterplot(data = under240, x = "turns", y = "chess_time", color = BLUE, s = size2, linewidth = line_width, edgecolor = WHITE)
plt.title("Turns & Chess Time (Under 240 minutes)", fontsize = title_size)
plt.xlabel("Number of turns")
plt.ylabel("Minutes")
plt.show()
    # Black rating and white rating
fig = plt.figure(figsize = (10,10))
sb.scatterplot(data = df, x = "black_rating", y = "white_rating", color = BLACK, s = size1, linewidth = .75, edgecolor = BLACK, hue = "winner", palette = [WHITE, BLACK, RED])
plt.title("Black Rating vs White Rating", fontsize = title_size)
plt.xlabel("Black rating")
plt.ylabel("White rating")
plt.show()
    # Rating difference and turns
fig = plt.figure(figsize = (10,8))
sb.scatterplot(data = df, x = "rating_difference", y = "turns", size = "opening_ply", hue = "victory_status", sizes = (30,300), alpha = 0.5, linewidth = line_width, edgecolor = WHITE)
plt.title("Rating Difference & Turns", fontsize = title_size)
plt.xlabel("Rating difference")
plt.ylabel("Number of turns")
plt.show()

# DATA PREPROCESSING
# Data Formatting
time1 = 0       # 0 second
time2 = 166.67  # 10000 seconds
corrupted_time = df[(df["chess_time"] == time1) | (df["chess_time"] == time2)]["chess_time"].count()       # 9282 corrupted time data
df.drop(labels = ["created_at","last_move_at","increment_code","white_id","black_id","moves","opening_name","chess_time"], axis = 1, inplace = True)

# Data Cleaning
nan_count = df.isnull().sum().sum()                     # 0 -> imputation does not needed
duplicated_rows = df[df["id"].duplicated()].count()     # 945 duplicated rows

df.drop_duplicates(subset = ["id"], inplace = True)     # Removing duplicated rows
df.reset_index(drop = True, inplace = True)             # Reindexing

# Encoding
ohe = OneHotEncoder()       # One-hot encoding method is used for "victory_status" feature
le = LabelEncoder()         # Label encoding method is used for "opening_eco", "rated" and "winner" features
    # "rated" encoding
rated_encoding = np.where(df["rated"] == True, 1, 0)     # Binary -> True = 1, False = 0
df_rated = pd.DataFrame(data = rated_encoding, columns = ["rated_or_not"])
    # "victory_status" encoding
victory_status = ["vic_draw","vic_mate","vic_outoftime","vic_resign"]
vic_encoding = ohe.fit_transform(df.iloc[:,3:4].values).toarray()
df_vic = pd.DataFrame(data = vic_encoding, columns = victory_status)
    # "opening_eco" encoding
open_eco = le.fit_transform(df.iloc[:,7])
df_eco = pd.DataFrame(data = open_eco, columns = ["eco"])

df.drop(labels = ["rated","victory_status"], axis = 1, inplace = True)     # Drop original columns
df = pd.concat([df,df_rated,df_vic,df_eco], axis = 1)                      # Merge results with dataframe

# FEATURE SELECTION/EXTRACTION - Anova F-value method
df_x = pd.concat([df.iloc[:,1:2],df.iloc[:,3:5],df.iloc[:,6:]], axis = 1)
df_y = df["winner"]
x = df_x.iloc[:,:].values   # independent variables
y = df_y.iloc[:].values     # dependent variable

anova_method = SelectKBest(f_classif, k = 8)    # 8 features out of 11
x_kbest = anova_method.fit_transform(x, y)

# DATASET SPLITTING
x_train, x_test, y_train, y_test = train_test_split(x_kbest, y, test_size = 0.33, random_state = 1)

# DATA SCALING - Standart Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# MODEL TRAINING AND EVALUATION - Classification
def conf_mx_visualization(y_pred, title, matrix_color):
     conf_mx = confusion_matrix(y_test, y_pred)
    
     counts = [count for count in conf_mx.flatten()]
     percentages = ["%.2f%%" % percentage for percentage in (conf_mx.flatten() / np.sum(conf_mx) * 100)]
     matrix_values = [f"{count}\n{percentage}" for count,percentage in zip(counts, percentages)]
     matrix_values= np.asarray(matrix_values).reshape(3,3)

     plt.figure(figsize = (10,8))
     axs = sb.heatmap(conf_mx, annot = matrix_values, fmt = "", cmap = matrix_color)
     axs.xaxis.set_ticklabels(["Black","Draw","White"])
     axs.yaxis.set_ticklabels(["Black","Draw","White"])
     plt.title(title, fontsize = conf_title)    
     plt.xlabel("Predicted", fontsize = title_size)
     plt.ylabel("Actual", fontsize = title_size)
     plt.show()

def evaluation_report(y_pred):
    print(f"Accuracy score: {round(accuracy_score(y_test, y_pred),5)}")
    print(f'Precision score: {round(precision_score(y_test, y_pred, average = "macro"),5)}')
    print(f'Sensivity score: {round(recall_score(y_test, y_pred, average = "macro"),5)}')
    print(f'F1 score: {round(f1_score(y_test, y_pred, average = "macro"),5)}\n')

# Logistic Regression
log_reg = LogisticRegression(max_iter = 125)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
title = "Logistic Regression Confusion Matrix"
matrix_color = "Purples"

conf_mx_visualization(y_pred_log, title, matrix_color)
print("Logistic Regression")
evaluation_report(y_pred_log)
# K-NN
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
title = "K-Nearest Neighbors Confusion Matrix"
matrix_color = "Blues"

conf_mx_visualization(y_pred_knn, title, matrix_color)
print("K-NN")
evaluation_report(y_pred_knn)
# SVM
    # Linear Kernel
svc = SVC(kernel = "linear")
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
title = "Support Vector Machine (linear) Confusion Matrix"
matrix_color = "Greens"

conf_mx_visualization(y_pred_svc, title, matrix_color)
print("SVM (linear)")
evaluation_report(y_pred_svc)
    # RBF Kernel
svc = SVC(kernel = "rbf")
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
title = "Support Vector Machine (RBF) Confusion Matrix"
matrix_color = "YlOrRd"

conf_mx_visualization(y_pred_svc, title, matrix_color)
print("SVM (RBF)")
evaluation_report(y_pred_svc)
    # Polynomial Kernel
svc = SVC(kernel = "poly")
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
title = "Support Vector Machine (polynomial) Confusion Matrix"
matrix_color = "Greys"

conf_mx_visualization(y_pred_svc, title, matrix_color)
print("SVM (polinomial)")
evaluation_report(y_pred_svc)
# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
title = "Naive Bayes Confusion Matrix"
matrix_color = "GnBu"

conf_mx_visualization(y_pred_nb, title, matrix_color)
print("Naive Bayes")
evaluation_report(y_pred_nb)
# Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
title = "Decision Tree Confusion Matrix"
matrix_color = "Reds"

conf_mx_visualization(y_pred_dtree, title, matrix_color)
print("Decision Tree")
evaluation_report(y_pred_dtree)
# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
title = "Random Forest Confusion Matrix"
matrix_color = "PuBuGn"

conf_mx_visualization(y_pred_rf, title, matrix_color)
print("Random Forest")
evaluation_report(y_pred_rf)