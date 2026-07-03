import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from "react";
import { useOnlineStatus } from "../hooks/useOnlineStatus";

export interface ConnectionError {
  code: string;
  message: string;
}

interface ConnectionContextValue {
  networkOnline: boolean;
  backendReachable: boolean;
  wsConnected: boolean;
  activeSessionId: string | null;
  lastError: ConnectionError | null;
  setWsConnected: (v: boolean) => void;
  setActiveSessionId: (id: string | null) => void;
  setLastError: (err: ConnectionError | null) => void;
  clearError: () => void;
}

const ConnectionContext = createContext<ConnectionContextValue>({
  networkOnline: true,
  backendReachable: true,
  wsConnected: false,
  activeSessionId: null,
  lastError: null,
  setWsConnected: () => {},
  setActiveSessionId: () => {},
  setLastError: () => {},
  clearError: () => {},
});

interface Props {
  children: ReactNode;
  backendReachable: boolean;
}

export function ConnectionProvider({ children, backendReachable }: Props) {
  const networkOnline = useOnlineStatus();
  const [wsConnected, setWsConnected] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [lastError, setLastError] = useState<ConnectionError | null>(null);

  const clearError = useCallback(() => setLastError(null), []);

  return (
    <ConnectionContext.Provider
      value={{
        networkOnline,
        backendReachable,
        wsConnected,
        activeSessionId,
        lastError,
        setWsConnected,
        setActiveSessionId,
        setLastError,
        clearError,
      }}
    >
      {children}
    </ConnectionContext.Provider>
  );
}

export function useConnection() {
  return useContext(ConnectionContext);
}
