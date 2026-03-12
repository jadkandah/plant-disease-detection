import { useState, useEffect } from 'react';
import * as Location from 'expo-location';

// Free-tier OpenWeatherMap key — replace with your own for production or use .env file
const OPENWEATHERMAP_API_KEY = process.env.EXPO_PUBLIC_WEATHER_API_KEY || 'YOUR_OPENWEATHER_API_KEY';

export interface WeatherData {
  temperature: number;
  humidity: number;
  description: string;
  icon: string;
  windSpeed: number;
  feelsLike: number;
  pressure: number;
  riskLevel: 'low' | 'medium' | 'high';
  riskMessage: string;
  cityName: string;
}

export interface LocationData {
  latitude: number;
  longitude: number;
  cityName: string;
  country: string;
}

/**
 * Estimates disease risk based on weather conditions.
 * High humidity + warm temperatures = higher fungal disease risk.
 */
function estimateRisk(temp: number, humidity: number): { level: 'low' | 'medium' | 'high'; message: string } {
  if (humidity > 80 && temp > 20 && temp < 35) {
    return { level: 'high', message: 'High humidity & warm temps increase fungal disease risk. Inspect crops closely.' };
  }
  if (humidity > 60 && temp > 15) {
    return { level: 'medium', message: 'Moderate conditions. Monitor your crops for early signs of disease.' };
  }
  return { level: 'low', message: 'Current weather conditions are favorable for healthy crops.' };
}

/**
 * Custom hook that fetches REAL weather data based on GPS location
 * and estimates disease risk level. Also exposes location data.
 */
export function useWeatherRisk() {
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [location, setLocation] = useState<LocationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchWeather();
  }, []);

  const fetchWeather = async () => {
    try {
      setLoading(true);
      setError(null);

      // Request location permission
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setError('Location permission not granted');
        setLoading(false);
        return;
      }

      const loc = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced,
      });

      const { latitude, longitude } = loc.coords;

      // Fetch real weather from OpenWeatherMap
      const response = await fetch(
        `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=${OPENWEATHERMAP_API_KEY}`
      );

      if (!response.ok) {
        throw new Error(`Weather API error: ${response.status}`);
      }

      const data = await response.json();
      const temp = data.main.temp;
      const humidity = data.main.humidity;
      const risk = estimateRisk(temp, humidity);
      const cityName = data.name || 'Unknown';
      const country = data.sys?.country || '';

      // Set location data for ProfileScreen
      setLocation({
        latitude,
        longitude,
        cityName,
        country,
      });

      // Set weather data
      setWeather({
        temperature: Math.round(temp),
        humidity,
        description: data.weather[0]?.description || 'Clear',
        icon: data.weather[0]?.icon || '01d',
        windSpeed: data.wind?.speed || 0,
        feelsLike: Math.round(data.main?.feels_like || temp),
        pressure: data.main?.pressure || 0,
        riskLevel: risk.level,
        riskMessage: risk.message,
        cityName,
      });
    } catch (err: any) {
      console.log('Weather fetch error:', err);
      setError(err.message || 'Failed to fetch weather');
    } finally {
      setLoading(false);
    }
  };

  return { weather, location, loading, error, refresh: fetchWeather };
}
