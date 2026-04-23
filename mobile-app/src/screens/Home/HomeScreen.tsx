import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { HeartPulse, Camera, Image as ImageIcon, CloudRain, Shield, AlertTriangle, Wifi, WifiOff, Cloud, CloudOff } from 'lucide-react-native';
import { useWeatherRisk } from '../../services/weather/useWeatherRisk';
import { useTranslation } from '../../store/LanguageContext';
import { useNetworkStatus } from '../../services/network/useNetworkStatus';
import { useModelMode } from '../../store/ModelModeContext';

const riskColors = {
  low: { bg: '#E8F5E9', border: '#C8E6C9', text: '#2E7D32', icon: Shield },
  medium: { bg: '#FFF3E0', border: '#FFE0B2', text: '#E65100', icon: CloudRain },
  high: { bg: '#FFEBEE', border: '#FFCDD2', text: '#C62828', icon: AlertTriangle },
};

export default function HomeScreen({ navigation }: any) {
  const { weather, loading: weatherLoading } = useWeatherRisk();
  const { t, isRTL } = useTranslation();
  const { isConnected } = useNetworkStatus();
  const { isOnlineMode } = useModelMode();

  const risk = weather ? riskColors[weather.riskLevel] : riskColors.low;
  const RiskIcon = risk.icon;

  const riskLabel = weather
    ? weather.riskLevel === 'high' ? t('home.highRisk')
    : weather.riskLevel === 'medium' ? t('home.mediumRisk')
    : t('home.lowRisk')
    : '';

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Connection Status Bar */}
        <View style={[styles.connectionBar, isConnected ? styles.connectedBar : styles.disconnectedBar, isRTL && styles.rtlRow]}>
          {isConnected ? <Wifi color="#2E7D32" size={16} /> : <WifiOff color="#C62828" size={16} />}
          <Text style={[styles.connectionText, { color: isConnected ? '#2E7D32' : '#C62828' }]}>
            {isConnected ? (isRTL ? 'متصل بالإنترنت' : 'Connected to Internet') : (isRTL ? 'غير متصل بالإنترنت' : 'No Internet Connection')}
          </Text>
        </View>

        {/* Model Mode Indicator */}
        <View style={[styles.connectionBar, isOnlineMode ? styles.onlineModeBar : styles.offlineModeBar, isRTL && styles.rtlRow]}>
          {isOnlineMode ? <Cloud color="#1565C0" size={16} /> : <CloudOff color="#E65100" size={16} />}
          <Text style={[styles.connectionText, { color: isOnlineMode ? '#1565C0' : '#E65100' }]}>
            {isOnlineMode ? t('home.modelOnline') : t('home.modelOffline')}
          </Text>
        </View>

        <View style={styles.header}>
          <Text style={[styles.greeting, isRTL && styles.rtlText]}>{t('home.greeting')}</Text>
          <Text style={[styles.title, isRTL && styles.rtlText]}>{t('home.title')}</Text>
        </View>

        {/* Weather Risk Card — Real Data */}
        <View style={[styles.weatherCard, { backgroundColor: risk.bg, borderColor: risk.border }]}>
          {weatherLoading ? (
            <ActivityIndicator color="#2E7D32" />
          ) : weather ? (
            <>
              <View style={[styles.weatherHeader, isRTL && styles.rtlRow]}>
                <RiskIcon color={risk.text} size={28} />
                <View style={[styles.weatherInfo, isRTL && { marginRight: 12, marginLeft: 0 }]}>
                  <Text style={[styles.weatherRisk, { color: risk.text }]}>{riskLabel}</Text>
                  <Text style={styles.weatherCity}>{weather.cityName}</Text>
                </View>
              </View>

              {/* Real weather stats row */}
              <View style={[styles.statsRow, isRTL && styles.rtlRow]}>
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{weather.temperature}°C</Text>
                  <Text style={styles.statLabel}>{isRTL ? 'الحرارة' : 'Temp'}</Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{weather.humidity}%</Text>
                  <Text style={styles.statLabel}>{isRTL ? 'الرطوبة' : 'Humidity'}</Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{weather.feelsLike}°C</Text>
                  <Text style={styles.statLabel}>{isRTL ? 'الإحساس' : 'Feels'}</Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{weather.windSpeed} m/s</Text>
                  <Text style={styles.statLabel}>{isRTL ? 'الرياح' : 'Wind'}</Text>
                </View>
              </View>

              <Text style={[styles.weatherDesc, { textTransform: 'capitalize' }]}>{weather.description}</Text>
              <Text style={[styles.weatherMessage, isRTL && styles.rtlText]}>{weather.riskMessage}</Text>
            </>
          ) : (
            <Text style={styles.weatherMessage}>{t('weather.unavailable')}</Text>
          )}
        </View>

        {/* Action Cards */}
        <View style={[styles.actionContainer, isRTL && styles.rtlRow]}>
          <TouchableOpacity style={[styles.actionCard, { backgroundColor: '#E8F5E9' }]} onPress={() => navigation.navigate('Camera')}>
            <Camera color="#2E7D32" size={32} />
            <Text style={styles.cardTitle}>{t('home.takePhoto')}</Text>
            <Text style={styles.cardSubtitle}>{t('home.cameraSubtitle')}</Text>
          </TouchableOpacity>

          <TouchableOpacity style={[styles.actionCard, { backgroundColor: '#F3E5F5' }]} onPress={() => navigation.navigate('Gallery')}>
            <ImageIcon color="#6A1B9A" size={32} />
            <Text style={styles.cardTitle}>{t('home.uploadGallery')}</Text>
            <Text style={styles.cardSubtitle}>{t('home.gallerySubtitle')}</Text>
          </TouchableOpacity>
        </View>

        {/* Daily Tip */}
        <View style={[styles.tipContainer, isRTL && styles.rtlRow]}>
          <HeartPulse color="#E65100" size={24} style={isRTL ? { marginLeft: 16 } : { marginRight: 16 }} />
          <View style={styles.tipTextContainer}>
            <Text style={[styles.tipTitle, isRTL && styles.rtlText]}>{t('home.dailyTip')}</Text>
            <Text style={[styles.tipText, isRTL && styles.rtlText]}>{t('home.tipText')}</Text>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  scrollContent: { padding: 20 },
  connectionBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 10,
    marginBottom: 16,
    gap: 8,
  },
  connectedBar: { backgroundColor: '#E8F5E9' },
  disconnectedBar: { backgroundColor: '#FFEBEE' },
  onlineModeBar: { backgroundColor: '#E3F2FD' },
  offlineModeBar: { backgroundColor: '#FFF3E0' },
  connectionText: { fontSize: 13, fontWeight: '600' },
  header: { marginBottom: 20 },
  greeting: { fontSize: 16, color: '#666' },
  title: { fontSize: 28, fontWeight: 'bold', color: '#2E7D32', marginTop: 5 },
  rtlText: { textAlign: 'right' },
  rtlRow: { flexDirection: 'row-reverse' },
  weatherCard: { borderRadius: 16, padding: 16, marginBottom: 24, borderWidth: 1 },
  weatherHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
  weatherInfo: { marginLeft: 12 },
  weatherRisk: { fontSize: 14, fontWeight: 'bold', letterSpacing: 1 },
  weatherCity: { fontSize: 13, color: '#666', marginTop: 2 },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: 'rgba(255,255,255,0.6)',
    borderRadius: 12,
    paddingVertical: 12,
    marginBottom: 12,
  },
  statItem: { alignItems: 'center', flex: 1 },
  statValue: { fontSize: 16, fontWeight: 'bold', color: '#333' },
  statLabel: { fontSize: 11, color: '#777', marginTop: 2 },
  statDivider: { width: 1, backgroundColor: 'rgba(0,0,0,0.1)' },
  weatherDesc: { fontSize: 14, color: '#555', marginBottom: 6 },
  weatherMessage: { fontSize: 14, color: '#444', lineHeight: 20 },
  actionContainer: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 24 },
  actionCard: {
    width: '48%', padding: 20, borderRadius: 16, alignItems: 'center', justifyContent: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 4, elevation: 3,
  },
  cardTitle: { fontSize: 16, fontWeight: 'bold', color: '#333', marginTop: 12, textAlign: 'center' },
  cardSubtitle: { fontSize: 12, color: '#666', marginTop: 4, textAlign: 'center' },
  tipContainer: { backgroundColor: '#FFF3E0', borderRadius: 12, padding: 16, flexDirection: 'row', alignItems: 'center' },
  tipTextContainer: { flex: 1 },
  tipTitle: { fontSize: 16, fontWeight: 'bold', color: '#E65100', marginBottom: 4 },
  tipText: { fontSize: 14, color: '#333', lineHeight: 20 },
});
