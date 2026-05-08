import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAuth } from '../../store/AuthContext';
import { useTranslation } from '../../store/LanguageContext';

export default function ProfileScreen({ navigation }: any) {
  const { user, logout } = useAuth();
  const { t, isRTL } = useTranslation();

  return (
    <SafeAreaView style={styles.container}>
      <Text style={[styles.title, isRTL && styles.rtlText]}>{t('profile.title')}</Text>

      <View style={styles.infoCard}>
        <Text style={[styles.label, isRTL && styles.rtlText]}>{t('profile.name')}</Text>
        <Text style={[styles.value, isRTL && styles.rtlText]}>{user?.full_name || 'N/A'}</Text>

        <Text style={[styles.label, isRTL && styles.rtlText]}>{t('profile.email')}</Text>
        <Text style={[styles.value, isRTL && styles.rtlText]}>{user?.email || 'N/A'}</Text>

        <Text style={[styles.label, isRTL && styles.rtlText]}>{t('profile.phone')}</Text>
        <Text style={[styles.value, isRTL && styles.rtlText]}>{user?.phone_number || 'N/A'}</Text>
      </View>

      {user?.is_admin && (
        <TouchableOpacity style={styles.adminButton} onPress={() => navigation.navigate('AdminDashboard')}>
          <Text style={styles.adminButtonText}>{t('profile.adminDashboard')}</Text>
        </TouchableOpacity>
      )}

      <TouchableOpacity style={styles.logoutButton} onPress={logout}>
        <Text style={styles.logoutText}>{t('profile.logOut')}</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5', padding: 20 },
  title: { fontSize: 28, fontWeight: 'bold', color: '#2E7D32', marginBottom: 20 },
  rtlText: { textAlign: 'right' },
  rtlRow: { flexDirection: 'row-reverse' },
  infoCard: {
    backgroundColor: 'white', padding: 20, borderRadius: 12,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 4, elevation: 3,
    marginBottom: 16,
  },
  label: { fontSize: 14, color: '#666', marginTop: 10 },
  value: { fontSize: 18, color: '#333', fontWeight: '500', marginBottom: 5 },
  logoutButton: { backgroundColor: '#E53935', padding: 15, borderRadius: 8, alignItems: 'center' },
  logoutText: { color: 'white', fontSize: 16, fontWeight: 'bold' },
  adminButton: { backgroundColor: '#1565C0', padding: 15, borderRadius: 8, alignItems: 'center', marginBottom: 12 },
  adminButtonText: { color: 'white', fontSize: 16, fontWeight: 'bold' },
});
