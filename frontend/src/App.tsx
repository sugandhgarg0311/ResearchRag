import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Papers from "./pages/Papers";
import Chat from "./pages/Chat";
import PaperDetail from "./pages/PaperDetail";
import Ask from "./pages/Ask";
import Retrieve from "./pages/Retrieve";
import Specs from "./pages/Specs";
import SpecDetail from "./pages/SpecDetail";
import Implementations from "./pages/Implementations";
import ImplementationDetail from "./pages/ImplementationDetail";
import Reproduction from "./pages/Reproduction";
import Evaluations from "./pages/Evaluations";
import Experiments from "./pages/Experiments";
import Jobs from "./pages/Jobs";
import Health from "./pages/Health";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="papers" element={<Papers />} />
        <Route path="chat" element={<Chat />} />
        <Route path="papers/:paperId" element={<PaperDetail />} />
        <Route path="papers/:paperId/ask" element={<Ask />} />
        <Route path="papers/:paperId/retrieve" element={<Retrieve />} />
        <Route path="papers/:paperId/specs" element={<Specs />} />
        <Route path="papers/:paperId/implementations" element={<Implementations />} />
        <Route path="papers/:paperId/reproduction" element={<Reproduction />} />
        <Route path="papers/:paperId/evaluations" element={<Evaluations />} />
        <Route path="papers/:paperId/experiments" element={<Experiments />} />
        <Route path="specs/:specId" element={<SpecDetail />} />
        <Route path="implementations/:implId" element={<ImplementationDetail />} />
        <Route path="jobs" element={<Jobs />} />
        <Route path="health" element={<Health />} />
        <Route path="*" element={<Dashboard />} />
      </Route>
    </Routes>
  );
}
