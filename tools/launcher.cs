using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace LugonPokerLab
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            try
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string extraArgs = args.Length > 0 ? " " + string.Join(" ", args) : "";

                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = "pyw.exe";
                psi.Arguments = "-m lugon_poker.companion_gui" + extraArgs;
                psi.WorkingDirectory = baseDir;
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;

                try
                {
                    Process.Start(psi);
                }
                catch
                {
                    psi.FileName = "py.exe";
                    Process.Start(psi);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Could not start Lugon Poker Lab:\n" + ex.Message,
                    "Lugon Poker Lab",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
        }
    }
}
