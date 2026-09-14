import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { ViewModeProvider } from "./view";
import { ThemeProvider } from "./theme";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <ThemeProvider>
        <ViewModeProvider>
          <App />
        </ViewModeProvider>
      </ThemeProvider>
    </BrowserRouter>
  </React.StrictMode>
);
