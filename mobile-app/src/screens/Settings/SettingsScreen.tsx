import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ChevronRight, HelpCircle, Info, Cloud, CloudOff } from 'lucide-react-native';
import { useTranslation } from '../../store/LanguageContext';
import { useModelMode } from '../../store/ModelModeContext';

export default function SettingsScreen() {
  const { t, language, setLanguage, isRTL } = useTranslation();
  const { setModelMode, isOnlineMode, canUseOnlineMode } = useModelMode();

  const handleHelpSupport = () => {
    Alert.alert(
      t('settings.helpSupport'),
      isRTL ? 'تواصل معنا عبر البريد الإلكتروني: support@plantdisease.com' : 'Contact us via email at: support@plantdisease.com'
    );
  };

  const handleAbout = () => {
    Alert.alert(
      t('settings.about'),
      `${t('settings.version')}\n${t('settings.gradProject')}`
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text style={[styles.title, isRTL && styles.rtlText]}>{t('settings.title')}</Text>
      <ScrollView>

        <View style={styles.modelModeCard}>
          <View style={[styles.modelModeHeader, isRTL && styles.rtlRow]}>
            {isOnlineMode ? <Cloud color="#2E7D32" size={22} /> : <CloudOff color="#C62828" size={22} />}
            <Text style={[styles.modelModeTitle, isRTL && { marginRight: 10, marginLeft: 0 }]}>
              {t('settings.modelMode')}
            </Text>
          </View>
          <Text style={[styles.modelModeDesc, isRTL && styles.rtlText]}>
            {!canUseOnlineMode
              ? t('settings.onlineUnavailableDesc')
              : isOnlineMode
                ? t('settings.onlineModelDesc')
                : t('settings.offlineModelDesc')}
          </Text>
          <View style={[styles.modelToggle, isRTL && styles.rtlRow]}>
            <TouchableOpacity
              style={[styles.modeBtn, !canUseOnlineMode && styles.modeBtnDisabled, isOnlineMode && styles.modeBtnActiveOnline]}
              onPress={() => setModelMode('online')}
              disabled={!canUseOnlineMode}
            >
              <Cloud color={isOnlineMode ? 'white' : canUseOnlineMode ? '#666' : '#aaa'} size={16} />
              <Text style={[styles.modeBtnText, !canUseOnlineMode && styles.modeBtnTextDisabled, isOnlineMode && styles.modeBtnTextActive]}>
                {t('settings.onlineModel')}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.modeBtn, !canUseOnlineMode && styles.modeBtnDisabled, !isOnlineMode && styles.modeBtnActiveOffline]}
              onPress={() => setModelMode('offline')}
              disabled={!canUseOnlineMode}
            >
              <CloudOff color={!isOnlineMode ? 'white' : '#666'} size={16} />
              <Text style={[styles.modeBtnText, !canUseOnlineMode && isOnlineMode && styles.modeBtnTextDisabled, !isOnlineMode && styles.modeBtnTextActive]}>
                {t('settings.offlineModel')}
              </Text>
            </TouchableOpacity>
          </View>
        </View>


        <View style={[styles.settingRow, isRTL && styles.rtlRow]}>
          <Text style={styles.settingLabel}>{t('settings.language')}</Text>
          <View style={[styles.languageToggle, isRTL && styles.rtlRow]}>
            <TouchableOpacity
              style={[styles.langBtn, language === 'en' && styles.langBtnActive]}
              onPress={() => setLanguage('en')}
            >
              <Text style={[styles.langText, language === 'en' && styles.langTextActive]}>EN</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.langBtn, language === 'ar' && styles.langBtnActive]}
              onPress={() => setLanguage('ar')}
            >
              <Text style={[styles.langText, language === 'ar' && styles.langTextActive]}>AR</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.actionGroup}>
          <TouchableOpacity style={[styles.actionRow, isRTL && styles.rtlRow]} onPress={handleHelpSupport}>
            <View style={[styles.actionLeft, isRTL && styles.rtlRow]}>
              <HelpCircle color="#2E7D32" size={24} />
              <Text style={[styles.actionLabel, isRTL ? { marginRight: 12 } : { marginLeft: 12 }]}>{t('settings.helpSupport')}</Text>
            </View>
            <ChevronRight color="#ccc" size={20} style={isRTL && styles.rtlIcon} />
          </TouchableOpacity>

          <View style={styles.divider} />

          <TouchableOpacity style={[styles.actionRow, isRTL && styles.rtlRow]} onPress={handleAbout}>
            <View style={[styles.actionLeft, isRTL && styles.rtlRow]}>
              <Info color="#2E7D32" size={24} />
              <Text style={[styles.actionLabel, isRTL ? { marginRight: 12 } : { marginLeft: 12 }]}>{t('settings.about')}</Text>
            </View>
            <ChevronRight color="#ccc" size={20} style={isRTL && styles.rtlIcon} />
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5', padding: 20 },
  title: { fontSize: 28, fontWeight: 'bold', color: '#2E7D32', marginBottom: 20 },
  rtlText: { textAlign: 'right' },
  rtlRow: { flexDirection: 'row-reverse' },
  rtlIcon: { transform: [{ scaleX: -1 }] },
  settingRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    backgroundColor: 'white', padding: 16, borderRadius: 12, marginBottom: 12,
    shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 4, elevation: 2,
  },
  settingLabel: { fontSize: 16, color: '#333' },

  // Model Mode Card
  modelModeCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  modelModeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  modelModeTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginLeft: 10,
  },
  modelModeDesc: {
    fontSize: 13,
    color: '#888',
    marginBottom: 14,
    lineHeight: 18,
  },
  modelToggle: {
    flexDirection: 'row',
    gap: 10,
  },
  modeBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: '#ddd',
    backgroundColor: '#FAFAFA',
  },
  modeBtnActiveOnline: {
    backgroundColor: '#2E7D32',
    borderColor: '#2E7D32',
  },
  modeBtnActiveOffline: {
    backgroundColor: '#C62828',
    borderColor: '#C62828',
  },
  modeBtnDisabled: {
    opacity: 0.65,
  },
  modeBtnText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
  },
  modeBtnTextActive: {
    color: 'white',
  },
  modeBtnTextDisabled: {
    color: '#aaa',
  },

  languageToggle: { flexDirection: 'row', gap: 8 },
  langBtn: { paddingVertical: 6, paddingHorizontal: 16, borderRadius: 6, borderWidth: 1, borderColor: '#ccc' },
  langBtnActive: { backgroundColor: '#2E7D32', borderColor: '#2E7D32' },
  langText: { fontSize: 14, color: '#666' },
  langTextActive: { color: 'white', fontWeight: 'bold' },
  actionGroup: {
    backgroundColor: 'white', borderRadius: 12, marginTop: 20, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 4, elevation: 2,
    overflow: 'hidden'
  },
  actionRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 16
  },
  actionLeft: {
    flexDirection: 'row', alignItems: 'center'
  },
  actionLabel: {
    fontSize: 16, color: '#333'
  },
  divider: {
    height: 1, backgroundColor: '#F0F0F0', marginHorizontal: 16
  }
});
