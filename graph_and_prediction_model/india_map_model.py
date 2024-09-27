import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


df = pd.read_excel(f"{os.getcwd()}\\graph_and_prediction_model\\data\\crime_data_final.xlsx")
# Cleaning Crime Data

# print(df.isnull().sum())

df.fillna(0,inplace=True)

# print(df.isnull().sum())

df['Total Crimes'] = df.drop(['State','District','State|District','Latitude','Longitude','Population','Year'],axis=1).sum(axis=1)
df['District'] = df['District'].str.lower()
df.sort_values(by=['Year'],inplace=True)

# india_states = state_map_data
india_districts = gpd.read_file(f"{os.getcwd()}\\graph_and_prediction_model\\data\\shape_fill_data\\DISTRICT_BOUNDARY.shp")
india_districts['DISTRICT'] = india_districts['DISTRICT'].str.lower()
india_districts['STATE'] = india_districts['DISTRICT'].str.lower()

normalized_df = df[['State','District','State|District','Latitude','Longitude','Population','Year']]
for col in df.columns:
    if col not in normalized_df:
        normalized_df[col]=((df[col]*100000)/df['Population']).apply(np.ceil)
        normalized_df.rename(columns={col: f'{col} per 100K people'}, inplace=True)

location_data = normalized_df[['District','Latitude','Longitude']]
location_data = location_data.groupby('District').first().reset_index()
district_wise_crimes = normalized_df.drop(['State','State|District','Latitude','Longitude','Population'],axis=1)

thresholds = {
    'safe': 0.006 * 100000,        # 0.6%
    'less safe': 0.012 * 100000,   # 1.2%
    'unsafe': 0.025 * 100000,      # 2.5%
}

# Function to categorize 'total'
def categorize_total(value):
    if value < thresholds['safe']:
        return 'safe'
    elif value < thresholds['less safe']:
        return 'less safe'
    elif value < thresholds['unsafe']:
        return 'unsafe'
    else:
        return 'highly unsafe'
        
# Grouping Data by Districts
# state_wise_crimes_mean_values = state_wise_crimes.groupby(['State', 'District'])['Total Crimes per 100K people'].mean().apply(np.ceil).reset_index()
district_wise_crimes_grouped = district_wise_crimes.groupby('District')['Total Crimes per 100K people'].mean().apply(np.ceil).reset_index()

latest_district_wise_crimes = district_wise_crimes[district_wise_crimes['Year']==2022]
latest_district_wise_crimes_with_location = latest_district_wise_crimes.merge(location_data,how='left',on='District')


district_wise_crimes_grouped['safety_category'] = district_wise_crimes_grouped['Total Crimes per 100K people'].apply(categorize_total)


# Mergin data with shape data
district_wise_crimes_location_merged = india_districts.merge(district_wise_crimes_grouped, how='left', left_on='DISTRICT', right_on='District')
district_wise_crimes_location_merged_cleaned = district_wise_crimes_location_merged.dropna(subset=['safety_category'])

# Define a color map for the categories
safety_colors = {
    'safe': 'green',
    'less safe': 'yellow',
    'unsafe': 'orange',
    'highly unsafe': 'red'
}

# Create a new column in the GeoDataFrame for the color category
district_wise_crimes_location_merged_cleaned['color'] = district_wise_crimes_location_merged_cleaned['safety_category'].map(safety_colors)


# # Plot the map with the state colors based on safety category
# fig, ax = plt.subplots(1, 1, figsize=(10, 12))

# # Plot each state and fill it with the appropriate color
# district_wise_crimes_location_merged_cleaned.boundary.plot(ax=ax, linewidth=1)
# district_wise_crimes_location_merged_cleaned.plot(ax=ax, color=district_wise_crimes_location_merged_cleaned['color'])

# # Add a title
# plt.title('India States by Safety Categories')

# # Create custom legend patches
# legend_labels = {
#     'safe': 'green',
#     'less safe': 'yellow',
#     'unsafe': 'orange',
#     'highly unsafe': 'red'
# }

# # Create legend handles based on safety categories
# legend_patches = [mpatches.Patch(color=color, label=label) for label, color in legend_labels.items()]

# # Add the legend to the plot
# plt.legend(handles=legend_patches, loc='lower left', title="Safety Categories")

# # Show the plot
# plt.show()

district_wise_crimes_location_merged_cleaned.to_file('india-district-map-with-safety-category.geojson', driver='GeoJSON')
latest_district_wise_crimes_with_location.to_json('latest_district_wise_crimes_with_location.json',orient='records',lines=True)
