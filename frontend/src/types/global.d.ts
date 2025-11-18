// global.d.ts
// Declare CSS & asset modules for the TypeScript language server (editor)
// so imports like `import styles from './Card.module.css'` don't error in VS Code.
// This file affects editor type checking only and does not change runtime behavior.

// CSS modules
declare module '*.module.css' {
  const classes: { readonly [key: string]: string };
  export default classes;
}
declare module '*.module.scss' {
  const classes: { readonly [key: string]: string };
  export default classes;
}
declare module '*.module.sass' {
  const classes: { readonly [key: string]: string };
  export default classes;
}
declare module '*.module.less' {
  const classes: { readonly [key: string]: string };
  export default classes;
}

// Non-module CSS (global imports)
declare module '*.css';
declare module '*.scss';
declare module '*.sass';
declare module '*.less';

// Static assets
declare module '*.svg' {
  const src: string;
  export default src;
}
declare module '*.png';
declare module '*.jpg';
declare module '*.jpeg';
declare module '*.gif';
