import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

export interface AccessibilityPreferences {
  readonly highContrast: boolean;
  readonly largeText: boolean;
  readonly setHighContrast: (enabled: boolean) => void;
  readonly setLargeText: (enabled: boolean) => void;
}

const STORAGE_KEY = 'indra:accessibility-preferences';
const HIGH_CONTRAST_CLASS = 'a11y-high-contrast';
const LARGE_TEXT_CLASS = 'a11y-large-text';

const AccessibilityPreferencesContext = createContext<AccessibilityPreferences | undefined>(
  undefined,
);

const applyClass = (className: string, enabled: boolean) => {
  if (typeof document === 'undefined') {
    return;
  }
  document.documentElement.classList.toggle(className, enabled);
};

const loadPreferences = () => {
  if (typeof window === 'undefined') {
    return { highContrast: false, largeText: false };
  }
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      const initial = { highContrast: false, largeText: false };
      applyClass(HIGH_CONTRAST_CLASS, initial.highContrast);
      applyClass(LARGE_TEXT_CLASS, initial.largeText);
      return initial;
    }
    const parsed = JSON.parse(stored) as Partial<AccessibilityPreferences>;
    const initial = {
      highContrast: Boolean(parsed.highContrast),
      largeText: Boolean(parsed.largeText),
    };
    applyClass(HIGH_CONTRAST_CLASS, initial.highContrast);
    applyClass(LARGE_TEXT_CLASS, initial.largeText);
    return initial;
  } catch {
    const fallback = { highContrast: false, largeText: false };
    applyClass(HIGH_CONTRAST_CLASS, fallback.highContrast);
    applyClass(LARGE_TEXT_CLASS, fallback.largeText);
    return fallback;
  }
};

export interface AccessibilityPreferencesProviderProps {
  readonly children: React.ReactNode;
}

export function AccessibilityPreferencesProvider({
  children,
}: AccessibilityPreferencesProviderProps) {
  const [prefs, setPrefs] = useState(() => loadPreferences());

  useEffect(() => {
    applyClass(HIGH_CONTRAST_CLASS, prefs.highContrast);
    applyClass(LARGE_TEXT_CLASS, prefs.largeText);
    if (typeof window !== 'undefined') {
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
      } catch {
        // ignore storage errors
      }
    }
  }, [prefs]);

  const setHighContrast = useCallback((enabled: boolean) => {
    setPrefs((previous) => ({
      ...previous,
      highContrast: enabled,
    }));
  }, []);

  const setLargeText = useCallback((enabled: boolean) => {
    setPrefs((previous) => ({
      ...previous,
      largeText: enabled,
    }));
  }, []);

  const value = useMemo<AccessibilityPreferences>(
    () => ({
      highContrast: prefs.highContrast,
      largeText: prefs.largeText,
      setHighContrast,
      setLargeText,
    }),
    [prefs.highContrast, prefs.largeText, setHighContrast, setLargeText],
  );

  return (
    <AccessibilityPreferencesContext.Provider value={value}>
      {children}
    </AccessibilityPreferencesContext.Provider>
  );
}

export function useAccessibilityPreferences(): AccessibilityPreferences {
  const context = useContext(AccessibilityPreferencesContext);
  if (!context) {
    throw new Error(
      'useAccessibilityPreferences must be used within an AccessibilityPreferencesProvider',
    );
  }
  return context;
}
