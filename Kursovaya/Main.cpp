#include "MyForm.h" 
using namespace System;
using namespace System::Windows::Forms;
[STAThreadAttribute]
void Main(array<System::String ^> ^args)
{
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);
	Kursovaya::MyForm mainForm;
	Application::Run(%mainForm);
}