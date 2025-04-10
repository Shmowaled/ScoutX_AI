import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. تحميل البيانات مع تخطي الأسطر غير الضرورية
file_path = r"C:\Users\SHAHAD\OneDrive\Desktop\ScoutX\sports-ai-env\olympics.csv"
df = pd.read_csv(file_path, skiprows=3)  # نتخطى أول 3 أسطر

# 2. تحديد أسماء الأعمدة بناءً على الإخراج الذي ظهر
df.columns = [
    'City', 'Edition', 'Sport', 'Discipline', 
    'Athlete', 'Country', 'Gender', 'Event', 
    'Event_gender', 'Medal'
]

# 3. تنظيف البيانات
# إزالة أي صفوف فارغة تماماً
df.dropna(how='all', inplace=True)

# 4. تحليل البيانات
print("أول 5 صفوف من البيانات بعد التنظيف:")
print(df.head())

print("\nمعلومات عن البيانات:")
print(df.info())

print("\nتوزيع الميداليات:")
print(df['Medal'].value_counts())

# 5. تصور البيانات
# توزيع الميداليات حسب الجنس
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='Medal', hue='Gender')
plt.title('Distribution of medals by gender')
plt.show()

# أكثر الدول فوزًا بالميداليات
top_countries = df['Country'].value_counts().head(10)
plt.figure(figsize=(12,6))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title("Top 10 countries by number of medals")
plt.xlabel("عدد الميداليات")
plt.show()

# 6. تحضير البيانات للنموذج
# بما أن البيانات لا تحتوي على العمر والطول والوزن
# سنستخدم الميزات المتاحة
features = ['Edition', 'Sport', 'Discipline', 'Gender', 'Event_gender']
target = 'Medal'

# ترميز المتغيرات الفئوية
le = LabelEncoder()
for col in features:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

df[target] = le.fit_transform(df[target].fillna('None'))  # None=0, Bronze=1, Silver=2, Gold=3

# 7. تقسيم البيانات
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. بناء وتدريب النموذج
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. تقييم النموذج
y_pred = model.predict(X_test)

print("\nتقرير التصنيف:")
print(classification_report(y_test, y_pred))

print("\nمصفوفة الارتباك:")
print(confusion_matrix(y_test, y_pred))

# 10. تفسير النموذج (اختياري)
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nأهمية الميزات:")
print(feature_importances)