import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router";
import "./index.css";
import Dashboard from "./pages/Dashboard";
import VideoDetail from "./pages/VideoDetail";
createRoot(document.getElementById("root")).render(_jsx(StrictMode, { children: _jsx(BrowserRouter, { children: _jsxs(Routes, { children: [_jsx(Route, { path: "/", element: _jsx(Dashboard, {}) }), _jsx(Route, { path: "/video/:videoId", element: _jsx(VideoDetail, {}) })] }) }) }));
