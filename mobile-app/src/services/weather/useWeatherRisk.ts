import { useState, useEffect } from 'react';
import * as Location from 'expo-location';

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
 * Map WMO weather codes to human-readable descriptions and icon codes.
 */
function getWeatherDescription(code: number): { description: string; icon: string } {
  if (code === 0) return { description: 'Clear sky', icon: '01d' };
  if (code <= 3) return { description: 'Partly cloudy', icon: '02d' };
  if (code <= 49) return { description: 'Foggy', icon: '50d' };
  if (code <= 59) return { description: 'Drizzle', icon: '09d' };
  if (code <= 69) return { description: 'Rain', icon: '10d' };
  if (code <= 79) return { description: 'Snow', icon: '13d' };
  if (code <= 99) return { description: 'Thunderstorm', icon: '11d' };
  return { description: 'Clear', icon: '01d' };
}

/**
 * Custom hook that fetches REAL weather data based on GPS location
 * using Open-Meteo API (free, no API key needed).
 * Estimates disease risk level and exposes location data.
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

      // Fetch weather from Open-Meteo (free, no API key needed)
      const response = await fetch(
        `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,apparent_temperature,surface_pressure,wind_speed_10m,weather_code`
      );

      if (!response.ok) {
        throw new Error(`Weather API error: ${response.status}`);
      }

      const data = await response.json();
      const current = data.current;
      const temp = current.temperature_2m;
      const humidity = current.relative_humidity_2m;
      const risk = estimateRisk(temp, humidity);
      const weatherInfo = getWeatherDescription(current.weather_code || 0);

      // Reverse geocode to get city name
      let cityName = 'Your Location';
      let country = '';
      try {
        const [geo] = await Location.reverseGeocodeAsync({ latitude, longitude });
        if (geo) {
          cityName = geo.city || geo.subregion || geo.region || 'Your Location';
          country = geo.isoCountryCode || '';
        }
      } catch (geoErr) {
        // Silently fall back to default city name
      }

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
        description: weatherInfo.description,
        icon: weatherInfo.icon,
        windSpeed: current.wind_speed_10m || 0,
        feelsLike: Math.round(current.apparent_temperature || temp),
        pressure: current.surface_pressure || 0,
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
