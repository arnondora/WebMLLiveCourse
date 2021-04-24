import pandas as pd

# Load CSV Files
temperature = pd.read_csv('dataset/temperature.csv')
pressure = pd.read_csv('dataset/pressure.csv')
humidity = pd.read_csv('dataset/humidity.csv')
wind_direction = pd.read_csv('dataset/wind_direction.csv')
wind_speed = pd.read_csv('dataset/wind_speed.csv')
weather_description = pd.read_csv('dataset/weather_description.csv')

# Extract data from Philadelphia only
temperature_extracted = temperature['Philadelphia']
pressure_extracted = pressure['Philadelphia']
humidity_extracted = humidity['Philadelphia']
wind_direction_extracted = wind_direction['Philadelphia']
wind_speed_extracted = wind_speed['Philadelphia']
weather_description_extracted = weather_description['Philadelphia']

complete_df = pd.DataFrame({'temperature': temperature_extracted, 'pressure' : pressure_extracted, 'humidity': humidity_extracted, 'wind_direction' : wind_direction_extracted, 'wind_speed': wind_speed_extracted, 'weather_condition': weather_description_extracted})
complete_df = complete_df.dropna()

# Rename to Rain/Not Rain Conditions
rain_conditions = ['light rain', 'heavy intensity rain', 'moderate rain', 'thunderstorm with rain', 'thunderstorm with light rain', 'proximity thunderstorm', 'very heavy rain', 'thunderstorm', 'thunderstorm with heavy rain', 'light rain and snow', 'light intensity shower rain', 'shower rain']

rain_data = pd.DataFrame()
for condition in rain_conditions :
    selected_data = complete_df[complete_df['weather_condition'] == condition]
    rain_data = pd.concat([rain_data, selected_data])

rain_data['weather_condition'] = 'rain'

selected_data = complete_df[complete_df['weather_condition'] == 'sky is clear']
clear_sky_data = pd.DataFrame()
clear_sky_data = pd.concat([selected_data])
clear_sky_data['weather_condition'] = 'sunshine'

complete_df = pd.concat([rain_data, clear_sky_data])
complete_df.to_csv('dataset/extracted_data.csv', index=False)
