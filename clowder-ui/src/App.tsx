import { useEffect } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { TopBar } from "./components/TopBar";
import { AgentTree } from "./components/AgentTree";
import { EventStream } from "./components/EventStream";
import { DetailPanel } from "./components/DetailPanel";
import { useStore } from "./store";

function HorizontalHandle() {
  return (
    <PanelResizeHandle
      style={{
        width: "5px",
        cursor: "col-resize",
        backgroundColor: "hsl(var(--border))",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
        <div style={{ width: "3px", height: "3px", borderRadius: "50%", backgroundColor: "hsl(var(--muted-foreground))", opacity: 0.4 }} />
        <div style={{ width: "3px", height: "3px", borderRadius: "50%", backgroundColor: "hsl(var(--muted-foreground))", opacity: 0.4 }} />
        <div style={{ width: "3px", height: "3px", borderRadius: "50%", backgroundColor: "hsl(var(--muted-foreground))", opacity: 0.4 }} />
      </div>
    </PanelResizeHandle>
  );
}

function VerticalHandle() {
  return (
    <PanelResizeHandle
      style={{
        height: "5px",
        cursor: "row-resize",
        backgroundColor: "hsl(var(--border))",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div style={{ display: "flex", gap: "3px" }}>
        <div style={{ width: "3px", height: "3px", borderRadius: "50%", backgroundColor: "hsl(var(--muted-foreground))", opacity: 0.4 }} />
        <div style={{ width: "3px", height: "3px", borderRadius: "50%", backgroundColor: "hsl(var(--muted-foreground))", opacity: 0.4 }} />
        <div style={{ width: "3px", height: "3px", borderRadius: "50%", backgroundColor: "hsl(var(--muted-foreground))", opacity: 0.4 }} />
      </div>
    </PanelResizeHandle>
  );
}

function MainContent() {
  return (
    <PanelGroup direction="vertical" style={{ height: "100%" }}>
      <Panel defaultSize={65} minSize={20}>
        <div style={{ height: "100%", overflow: "auto" }}>
          <AgentTree />
        </div>
      </Panel>
      <VerticalHandle />
      <Panel defaultSize={35} minSize={15}>
        <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
          <EventStream />
        </div>
      </Panel>
    </PanelGroup>
  );
}

function App() {
  const { connected, theme, selection } = useStore();

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  return (
    <div className="h-screen flex flex-col">
      <TopBar />
      {connected ? (
        <div className="flex-1 min-h-0">
          {selection ? (
            <PanelGroup direction="horizontal" style={{ height: "100%" }}>
              <Panel defaultSize={65} minSize={30}>
                <MainContent />
              </Panel>
              <HorizontalHandle />
              <Panel defaultSize={35} minSize={15}>
                <div style={{ height: "100%" }}>
                  <DetailPanel />
                </div>
              </Panel>
            </PanelGroup>
          ) : (
            <MainContent />
          )}
        </div>
      ) : (
        <div
          className="flex-1 flex items-center justify-center"
          style={{ color: "hsl(var(--muted-foreground))" }}
        >
          <p className="text-sm">
            Connect to a ClowderAgent to start monitoring
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
