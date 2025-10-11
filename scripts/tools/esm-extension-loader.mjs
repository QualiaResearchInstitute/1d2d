export async function resolve(specifier, context, defaultResolve) {
  const isRelative =
    specifier.startsWith('./') || specifier.startsWith('../') || specifier.startsWith('/');
  if (!isRelative) {
    return defaultResolve(specifier, context, defaultResolve);
  }

  try {
    return await defaultResolve(specifier, context, defaultResolve);
  } catch (error) {
    if (error && error.code === 'ERR_MODULE_NOT_FOUND' && !specifier.endsWith('.js')) {
      return defaultResolve(`${specifier}.js`, context, defaultResolve);
    }
    throw error;
  }
}
