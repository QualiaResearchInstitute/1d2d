import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import enMessages from './messages/en';
import esMessages from './messages/es';

export type Locale = 'en' | 'es';

export interface LocaleOption {
  readonly value: Locale;
  readonly label: string;
}

type MessageRecord = Record<string, string>;

const LOCALE_STORAGE_KEY = 'indra:locale';

const messagesByLocale: Record<Locale, MessageRecord> = {
  en: enMessages,
  es: esMessages,
};

const defaultLocale: Locale = 'en';

const localeOptions: readonly LocaleOption[] = [
  { value: 'en', label: enMessages['language.option.en'] },
  { value: 'es', label: esMessages['language.option.es'] },
];

const formatMessage = (template: string, values?: Record<string, unknown>) => {
  if (!values) {
    return template;
  }
  return template.replace(/\{(\w+)\}/g, (match, token) => {
    if (Object.prototype.hasOwnProperty.call(values, token)) {
      const value = values[token];
      return value === undefined || value === null ? '' : String(value);
    }
    return match;
  });
};

interface I18nContextValue {
  readonly locale: Locale;
  readonly setLocale: (next: Locale) => void;
  readonly t: (
    key: string,
    options?: { readonly values?: Record<string, unknown>; readonly fallback?: string },
  ) => string;
  readonly locales: readonly LocaleOption[];
}

const I18nContext = createContext<I18nContextValue | undefined>(undefined);

const detectInitialLocale = (): Locale => {
  if (typeof window !== 'undefined') {
    const stored = window.localStorage?.getItem(LOCALE_STORAGE_KEY);
    if (stored === 'en' || stored === 'es') {
      return stored;
    }
    const navigatorLocale = window.navigator?.language?.slice(0, 2).toLowerCase();
    if (navigatorLocale === 'es') {
      return 'es';
    }
  }
  return defaultLocale;
};

export interface LocalizationProviderProps {
  readonly children: React.ReactNode;
}

export function LocalizationProvider({ children }: LocalizationProviderProps) {
  const [locale, setLocaleState] = useState<Locale>(() => detectInitialLocale());

  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        window.localStorage?.setItem(LOCALE_STORAGE_KEY, locale);
      } catch {
        // ignore storage errors
      }
    }
  }, [locale]);

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
  }, []);

  const value = useMemo<I18nContextValue>(() => {
    const activeMessages = messagesByLocale[locale] ?? messagesByLocale[defaultLocale];
    const fallbackMessages = messagesByLocale[defaultLocale];
    const t: I18nContextValue['t'] = (key, options) => {
      const message = activeMessages[key] ?? fallbackMessages[key] ?? options?.fallback ?? key;
      return formatMessage(message, options?.values);
    };
    return {
      locale,
      setLocale,
      t,
      locales: localeOptions,
    };
  }, [locale, setLocale]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nContextValue {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useI18n must be used within a LocalizationProvider');
  }
  return context;
}
