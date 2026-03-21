import React, { useContext } from 'react';
import { View, ActivityIndicator } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Home, History, User, Settings } from 'lucide-react-native';
import { useAutoSync } from './src/services/sync/useAutoSync';
import { useTranslation } from './src/store/LanguageContext';

import { AuthProvider, AuthContext } from './src/store/AuthContext';
import { LanguageProvider } from './src/store/LanguageContext';
import { LoginScreen } from './src/screens/Auth/LoginScreen';
import { SignUpScreen } from './src/screens/Auth/SignUpScreen';

// Tab screens
import HomeScreen from './src/screens/Home/HomeScreen';
import HistoryScreen from './src/screens/History/HistoryScreen';
import ProfileScreen from './src/screens/Profile/ProfileScreen';
import SettingsScreen from './src/screens/Settings/SettingsScreen';

// Stack screens
import CameraScreen from './src/screens/Home/CameraScreen';
import GalleryScreen from './src/screens/Home/GalleryScreen';
import ResultScreen from './src/screens/Home/ResultScreen';
import AdminDashboardScreen from './src/screens/Admin/AdminDashboardScreen';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

const MainTabs = () => {
  const { t } = useTranslation();
  return (
    <Tab.Navigator
      screenOptions={{
        tabBarActiveTintColor: '#2E7D32',
        tabBarInactiveTintColor: 'gray',
        headerShown: false,
        tabBarStyle: {
          backgroundColor: '#fff',
          borderTopWidth: 1,
          borderTopColor: '#E0E0E0',
          paddingBottom: 5,
          paddingTop: 5,
          height: 60,
        },
      }}
    >
      <Tab.Screen
        name="HomeDashboard"
        component={HomeScreen}
        options={{ tabBarLabel: t('home.takePhoto').includes('Photo') ? 'Home' : 'الرئيسية', tabBarIcon: ({ color, size }) => <Home color={color} size={size} /> }}
      />
      <Tab.Screen
        name="HistoryTab"
        component={HistoryScreen}
        options={{ tabBarLabel: t('history.title').includes('History') ? 'History' : 'السجل', tabBarIcon: ({ color, size }) => <History color={color} size={size} /> }}
      />
      <Tab.Screen
        name="ProfileTab"
        component={ProfileScreen}
        options={{ tabBarLabel: t('profile.title').includes('Profile') ? 'Profile' : 'الملف', tabBarIcon: ({ color, size }) => <User color={color} size={size} /> }}
      />
      <Tab.Screen
        name="SettingsTab"
        component={SettingsScreen}
        options={{ tabBarLabel: t('settings.title').includes('Settings') ? 'Settings' : 'الإعدادات', tabBarIcon: ({ color, size }) => <Settings color={color} size={size} /> }}
      />
    </Tab.Navigator>
  );
};

const Navigation = () => {
  const { user, isLoading } = useContext(AuthContext);
  useAutoSync();

  if (isLoading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#fff' }}>
        <ActivityIndicator size="large" color="#2E7D32" />
      </View>
    );
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {user ? (
          <>
            <Stack.Screen name="MainTabs" component={MainTabs} />
            <Stack.Screen name="Camera" component={CameraScreen} />
            <Stack.Screen name="Gallery" component={GalleryScreen} />
            <Stack.Screen name="Result" component={ResultScreen} options={{ gestureEnabled: false }} />
            <Stack.Screen name="AdminDashboard" component={AdminDashboardScreen} />
          </>
        ) : (
          <>
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="SignUp" component={SignUpScreen} />
          </>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default function App() {
  return (
    <LanguageProvider>
      <AuthProvider>
        <Navigation />
      </AuthProvider>
    </LanguageProvider>
  );
}
