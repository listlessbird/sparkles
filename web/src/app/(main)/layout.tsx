import { QueryProvider } from "@/app/query-provider";
import { Header } from "@/components/nav/header";
import { HeaderMobile } from "@/components/nav/header-mobile";
import { SideNav } from "@/components/nav/sidenav";
import { SessionProvider } from "@/hooks/use-session";
import { validateRequest } from "@/lib/auth";

export default async function Layout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { user } = await validateRequest();

  return (
    <QueryProvider>
      <SessionProvider value={{ user: user }}>
        <div className="flex">
          <SideNav />
          <main className="min-h-screen flex-1">
            <Header />
            <HeaderMobile />
            <div className="space-y-8 max-w-2xl mx-auto pt-5 px-3 lg:pl-16">
              {children}
            </div>
          </main>
        </div>
      </SessionProvider>
    </QueryProvider>
  );
}
