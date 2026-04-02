import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    // Enable server actions and other App Router features
  },
  // Allow longer timeouts for Python pipeline calls
  serverExternalPackages: [],
};

export default nextConfig;
