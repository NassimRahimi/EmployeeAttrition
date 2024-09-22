import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objects as go
import matplotlib.colors
import streamlit as st
import warnings
import matplotlib
matplotlib.use('Agg')  # Prevent Matplotlib from trying to display plots interactively

# Additionally, filter Matplotlib specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')


# Function to plot sorted bar charts of satisfaction, monthly hours, evaluation, and time spent in company by department
def plot_sorted_department_metrics(df):
    fig, ax = plt.subplots(figsize=(20, 8))

    # Sorting values before plotting
    satisfaction_sorted = df.groupby('department')['satisfaction_level'].mean().sort_values(ascending=False)
    monthly_hours_sorted = df.groupby('department')['average_monthly_hours'].mean().sort_values(ascending=False)
    last_evaluation_sorted = df.groupby('department')['last_evaluation'].mean().sort_values(ascending=False)
    time_spent_sorted = df.groupby('department')['time_spend_company'].mean().sort_values(ascending=False)

    # Satisfaction Level
    ax = plt.subplot(2, 2, 1)
    sns.barplot(x=satisfaction_sorted.index, y=satisfaction_sorted.values, palette='Blues_d', ax=ax)
    ax.set_title('Average Satisfaction Level by Department', fontweight='bold')
    ax.set_xlabel('Department', fontweight='bold')
    ax.set_ylabel('Average Satisfaction Level', fontweight='bold')

    # Average Monthly Hours
    ax = plt.subplot(2, 2, 2)
    sns.barplot(x=monthly_hours_sorted.index, y=monthly_hours_sorted.values, palette='Reds_d', ax=ax)
    ax.set_title('Average Monthly Hours by Department', fontweight='bold')
    ax.set_xlabel('Department', fontweight='bold')
    ax.set_ylabel('Average Monthly Hours', fontweight='bold')

    # Last Evaluation
    ax = plt.subplot(2, 2, 3)
    sns.barplot(x=last_evaluation_sorted.index, y=last_evaluation_sorted.values, palette='Greens_d', ax=ax)
    ax.set_title('Average Last Evaluation by Department', fontweight='bold')
    ax.set_xlabel('Department', fontweight='bold')
    ax.set_ylabel('Average Last Evaluation', fontweight='bold')

    # Time Spent in Company
    ax = plt.subplot(2, 2, 4)
    sns.barplot(x=time_spent_sorted.index, y=time_spent_sorted.values, palette='Purples_d', ax=ax)
    ax.set_title('Average Time Spent in Company by Department', fontweight='bold')
    ax.set_xlabel('Department', fontweight='bold')
    ax.set_ylabel('Average Time Spent in Company (Years)', fontweight='bold')

    plt.tight_layout()
    return fig


# Function to plot pie charts for employees who left by department and salary
def plot_pie_charts_for_left(df):
    # Calculate left counts and percentages
    department_left_count = df[df['left'] == 1].groupby('department').size()
    salary_left_count = df[df['left'] == 1].groupby('salary').size()
    total_left = df[df['left'] == 1].shape[0]

    department_left_percentage_total = (department_left_count / total_left) * 100
    salary_left_percentage_total = (salary_left_count / total_left) * 100

    # Plot pie charts
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    # Pie chart for salary categories
    axs[0].pie(salary_left_percentage_total, labels=salary_left_percentage_total.index, autopct='%1.1f%%', 
               startangle=90, colors=sns.color_palette('Reds', len(salary_left_percentage_total)),
               textprops={'fontsize': 12, 'fontweight': 'bold'})
    axs[0].set_title('Proportion of Employees Left by Salary Category', fontsize=14, fontweight='bold')

    # Pie chart for departments
    axs[1].pie(department_left_percentage_total, labels=department_left_percentage_total.index, autopct='%1.1f%%', 
               startangle=90, colors=sns.color_palette('Blues', len(department_left_percentage_total)),
               textprops={'fontsize': 12, 'fontweight': 'bold'})
    axs[1].set_title('Proportion of Employees Left by Department', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


# Function to plot sorted bar charts of left percentages by department and salary
def plot_left_percentages(df):
    # Calculate left percentages
    department_left_percentage = df.groupby('department')['left'].mean() * 100
    salary_left_percentage = df.groupby('salary')['left'].mean() * 100

    department_left_percentage_sorted = department_left_percentage.sort_values(ascending=False)
    salary_left_percentage_sorted = salary_left_percentage.sort_values(ascending=False)

    # Plot bar charts
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))

    # Department left percentage (sorted)
    bars = axs[0].bar(department_left_percentage_sorted.index, department_left_percentage_sorted.values, color='skyblue')
    axs[0].set_title('Percentage of Employees Left by Department', fontsize=14, fontweight='bold')
    axs[0].set_xlabel('Department', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Percentage Left', fontsize=12, fontweight='bold')

    # Add values on top of bars
    for bar in bars:
        axs[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')

    # Salary left percentage (sorted)
    bars = axs[1].bar(salary_left_percentage_sorted.index, salary_left_percentage_sorted.values, color='salmon')
    axs[1].set_title('Percentage of Employees Left by Salary Category', fontsize=14, fontweight='bold')
    axs[1].set_xlabel('Salary', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Percentage Left', fontsize=12, fontweight='bold')

    # Add values on top of bars
    for bar in bars:
        axs[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig

# Function to plot distributions of numerical features
def plot_distributions(df):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    ax[0, 0].set_title('Distribution of Satisfaction Level')
    sns.histplot(df['satisfaction_level'], kde=True, color='blue', ax=ax[0, 0])

    ax[0, 1].set_title('Distribution of Last Evaluation')
    sns.histplot(df['last_evaluation'], kde=True, color='green', ax=ax[0, 1])

    ax[1, 0].set_title('Distribution of Average Monthly Hours')
    sns.histplot(df['average_monthly_hours'], kde=True, color='red', ax=ax[1, 0])

    ax[1, 1].set_title('Distribution of Time Spent in Company')
    sns.histplot(df['time_spend_company'], kde=True, color='purple', ax=ax[1, 1])

    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_numeric = numeric_df.corr()
    sns.heatmap(correlation_numeric, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap of Numerical Features')
    return fig

# Function to plot boxplots for numerical features vs 'left'
def plot_boxplots_vs_left(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    num_columns = len(numeric_columns)

    # Calculate rows needed
    nrows = (num_columns // 3) + (num_columns % 3 > 0)

    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(18, 4 * nrows))
    fig.suptitle('Boxplots of Numerical Features vs Left', fontsize=16)
    axes = axes.flatten()

    for i, column in enumerate(numeric_columns):
        sns.boxplot(x='left', y=column, data=df, ax=axes[i])
        axes[i].set_title(f'{column} vs Left')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_correlation_with_target(df):
    df_eda = df.copy()

    # Dropping the specified columns in df_eda while keeping df intact
    df_eda.drop(['salary', 'department'], axis=1, inplace=True)
    corr = df_eda.corr()
    corr = corr['left'].sort_values(ascending=False)[1:-1]

    pal = sns.color_palette("Reds_r", 135).as_hex()
    rgb = ['rgba'+str(matplotlib.colors.to_rgba(i, 0.7)) for i in pal]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=corr[corr >= 0], y=corr[corr >= 0].index, marker_color=rgb, orientation='h',
                         marker_line=dict(color=pal, width=2), name='',
                         hovertemplate='%{y} correlation with target: %{x:.3f}',
                         showlegend=False))
    
    pal = sns.color_palette("Blues", 100).as_hex()
    rgb = ['rgba' + str(matplotlib.colors.to_rgba(i, 0.7)) for i in pal]
    
    fig.add_trace(go.Bar(x=corr[corr < 0], y=corr[corr < 0].index, marker_color=rgb[25:], orientation='h',
                         marker_line=dict(color=pal[25:], width=2), name='',
                         hovertemplate='%{y} correlation with target: %{x:.3f}',
                         showlegend=False))
    
    fig.update_layout(
        title="Feature Correlations with Target",
        xaxis_title="Correlation",
        margin=dict(l=150),
        height=500,
        width=700,
        hovermode='closest',
        template=dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), height=500, width=1000))
    )
    
    return fig  # Returning the figure object



def plot_class_balance(data, target_column='left'):
    """
    Function to visualize the class balance of the target variable with bold fonts and numbers on bars.

    Parameters:
    - data: pandas DataFrame containing the dataset
    - target_column: the name of the target column to check class balance (default: 'left')
    """

    # Write the title for the section
    #st.write(f"### Class Balance of Target Variable (`{target_column}`)")

    # Create the figure for the count plot
    fig, ax = plt.subplots()

    # Plot the class balance with a countplot
    sns.countplot(x=target_column, data=data, ax=ax)

    # Add counts on top of the bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontweight='bold')

    # Set the title and axis labels with bold fonts
    #ax.set_title('Class Balance: Employees Stayed vs. Left', fontweight='bold')
    ax.set_xlabel('Employee Status', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')

    # Set tick labels to bold
    ax.xaxis.set_tick_params(labelsize=12, labelrotation=0, labelcolor='black', width=2)
    ax.yaxis.set_tick_params(labelsize=12, labelrotation=0, labelcolor='black', width=2)

    # Set fontweight for tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Render the plot with Streamlit
    st.pyplot(fig)

    # Calculate and display class percentages
    class_counts = data[target_column].value_counts(normalize=True) * 100
    st.write(f"Percentage of employees who stayed: {class_counts[0]:.2f}%")
    st.write(f"Percentage of employees who left: {class_counts[1]:.2f}%")