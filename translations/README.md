# Foreign language translations

The translations for German, French and Spanish are not considered stable yet 
and need proofreading.

If you want to add them back to the site, please do the following three steps:

- Copy the content of `all_locales.json` to `locales.json`.
- Copy the folders `de`, `fr` and `es` from this `translations` folder back into
  the `notebooks` folder.
- Then, copy the files `de.js`, `fr.js` and `es.js` from the `translations` folder
  into the `src/pages` folder.

If you then run `npm run dev` and visit the site, you should see the translations.
Deployment remains the same.