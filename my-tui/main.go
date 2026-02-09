package main

import (
	"fmt"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

func main() {
	app := tview.NewApplication()
	app.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		if event.Key() == tcell.KeyEscape {
			app.Stop()
		}
		return event
	})

	mainLayout := tview.NewFlex().
		SetDirection(tview.FlexRow)

	header := tview.NewTextView().
		SetText("Buggy - Terminal User Interface").
		SetTextAlign(tview.AlignCenter).
		SetTextColor(tcell.ColorTeal)

	menu := tview.NewList().
		SetSelectedTextColor(tcell.ColorTeal).
		SetSelectedBackgroundColor(tcell.ColorDarkGray)

	menu.AddItem("Dashboard", "System metrics and status", rune('1'), func() {
		app.SetRoot(createDashboard(app), true)
	})
	menu.AddItem("File Manager", "Browse files and directories", rune('2'), func() {
		app.SetRoot(createFileManager(app), true)
	})
	menu.AddItem("Text Editor", "Simple text editor", rune('3'), func() {
		app.SetRoot(createTextEditor(app), true)
	})
	menu.AddItem("Monitor", "Real-time system monitor", rune('4'), func() {
		app.SetRoot(createMonitor(app), true)
	})
	menu.AddItem("Form Wizard", "Multi-step form", rune('5'), func() {
		app.SetRoot(createFormWizard(app), true)
	})
	menu.AddItem("Exit", "Close Buggy", rune('q'), func() {
		app.Stop()
	})

	menu.SetBorder(true).SetTitle(" Menu ")

	content := tview.NewTextView().
		SetText("Use arrow keys to navigate, Enter to select, 'q' to go back").
		SetTextAlign(tview.AlignCenter).
		SetTextColor(tcell.ColorGray)

	footer := tview.NewTextView().
		SetText("Press 'q' to return to menu").
		SetTextAlign(tview.AlignCenter).
		SetTextColor(tcell.ColorGray)

	mainLayout.AddItem(header, 3, 0, false)
	mainLayout.AddItem(menu, 0, 1, true)
	mainLayout.AddItem(content, 3, 0, false)
	mainLayout.AddItem(footer, 1, 0, false)

	if err := app.SetRoot(mainLayout, true).EnableMouse(true).Run(); err != nil {
		panic(err)
	}
}

func createDashboard(app *tview.Application) *tview.Flex {
	flex := tview.NewFlex().SetDirection(tview.FlexRow)

	title := tview.NewTextView().
		SetText("Dashboard").
		SetTextAlign(tview.AlignCenter).
		SetTextColor(tcell.ColorTeal)

	metrics := []struct {
		title, value, status string
	}{
		{"CPU Usage", "45%", "OK"},
		{"Memory", "62%", "Warning"},
		{"Disk", "78%", "Critical"},
		{"Network", "120 Mbps", "OK"},
	}

	grid := tview.NewGrid().
		SetColumns(0, 0).
		SetRows(3, 3, 3, 3, 0).
		SetBorders(false)

	for i, m := range metrics {
		_ = tview.NewBox().
			SetBorder(true).
			SetTitle(m.title)

		text := tview.NewTextView().
			SetText(fmt.Sprintf("%s\n%s\n%s", m.title, m.value, m.status)).
			SetTextAlign(tview.AlignCenter)

		grid.AddItem(text, i, 0, 1, 1, 0, 0, false)
	}

	back := tview.NewButton("Back to Menu")
	back.SetLabelColor(tcell.ColorTeal)
	back.SetSelectedFunc(func() {
		app.SetRoot(createMainMenu(app), true)
	})

	flex.AddItem(title, 1, 0, false)
	flex.AddItem(grid, 0, 1, false)
	flex.AddItem(back, 1, 0, false)

	flex.SetBorderPadding(1, 1, 2, 2)

	return flex
}

func createFileManager(app *tview.Application) *tview.Flex {
	flex := tview.NewFlex().SetDirection(tview.FlexRow)

	title := tview.NewTextView().
		SetText("File Manager").
		SetTextAlign(tview.AlignCenter).
		SetTextColor(tcell.ColorTeal)

	files := tview.NewList().
		AddItem("document.txt", "", rune('d'), nil).
		AddItem("projects/", "", rune('p'), nil).
		AddItem("image.png", "", rune('i'), nil).
		AddItem("downloads/", "", rune('D'), nil).
		AddItem("config.yaml", "", rune('c'), nil).
		SetBorder(true).SetTitle(" Files ")

	back := tview.NewButton("Back to Menu")
	back.SetLabelColor(tcell.ColorTeal)
	back.SetSelectedFunc(func() {
		app.SetRoot(createMainMenu(app), true)
	})

	flex.AddItem(title, 1, 0, false)
	flex.AddItem(files, 0, 1, true)
	flex.AddItem(back, 1, 0, false)

	return flex
}

func createTextEditor(app *tview.Application) *tview.Flex {
	flex := tview.NewFlex().SetDirection(tview.FlexRow)

	title := tview.NewTextView().
		SetText("Text Editor").
		SetTextAlign(tview.AlignCenter).
		SetTextColor(tcell.ColorTeal)

	input := tview.NewTextArea().
		SetPlaceholder("Type here...")
	input.SetBorder(true).SetTitle(" Type ")

	help := tview.NewTextView().
		SetText("Type and navigate with arrow keys").
		SetTextAlign(tview.AlignCenter).
		SetTextColor(tcell.ColorGray)

	back := tview.NewButton("Back to Menu")
	back.SetLabelColor(tcell.ColorTeal)
	back.SetSelectedFunc(func() {
		app.SetRoot(createMainMenu(app), true)
	})

	flex.AddItem(title, 1, 0, false)
	flex.AddItem(input, 0, 1, true)
	flex.AddItem(help, 1, 0, false)
	flex.AddItem(back, 1, 0, false)

	return flex
}

func createMonitor(app *tview.Application) *tview.Flex {
	flex := tview.NewFlex().SetDirection(tview.FlexRow)

	title := tview.NewTextView().
		SetText("System Monitor").
		SetTextAlign(tview.AlignCenter).
		SetTextColor(tcell.ColorTeal)

	stats := tview.NewTextView().
		SetDynamicColors(true)

	go func() {
		for {
			app.QueueUpdateDraw(func() {
				cpu := 30 + (time.Now().Second() % 50)
				mem := 40 + (time.Now().Second() % 30)
				disk := 60 + (time.Now().Second() % 20)
				net := 100 - (time.Now().Second() % 50)

				stats.SetText(fmt.Sprintf(
					"CPU Usage: %d%% %s\nMemory Usage: %d%% %s\nDisk Usage: %d%% %s\nNetwork: %d Mbps %s\n\nUpdating every second...",
					cpu, bar(cpu),
					mem, bar(mem),
					disk, bar(disk),
					net, bar(net/10),
				))
			})
			time.Sleep(1 * time.Second)
		}
	}()

	stats.SetBorder(true).SetTitle(" Live Stats ")

	back := tview.NewButton("Back to Menu")
	back.SetLabelColor(tcell.ColorTeal)
	back.SetSelectedFunc(func() {
		app.SetRoot(createMainMenu(app), true)
	})

	flex.AddItem(title, 1, 0, false)
	flex.AddItem(stats, 0, 1, true)
	flex.AddItem(back, 1, 0, false)

	return flex
}

func createFormWizard(app *tview.Application) *tview.Flex {
	flex := tview.NewFlex().SetDirection(tview.FlexRow)

	title := tview.NewTextView().
		SetText("Form Wizard").
		SetTextAlign(tview.AlignCenter).
		SetTextColor(tcell.ColorTeal)

	form := tview.NewForm().
		SetFieldTextColor(tcell.ColorWhite).
		SetButtonTextColor(tcell.ColorBlack)

	form.AddInputField("Name", "", 20, nil, nil)
	form.AddInputField("Email", "", 20, nil, nil)
	form.AddInputField("Age", "", 20, nil, nil)
	form.AddDropDown("Country", []string{"USA", "UK", "Canada", "India", "Other"}, 0, nil)
	form.AddButton("Submit", func() {
		form.GetButton(0).SetLabel("Submitted!")
	})
	form.AddButton("Reset", func() {
		form.GetButton(0).SetLabel("Submit")
	})

	form.SetBorder(true).SetTitle(" Form ")

	back := tview.NewButton("Back to Menu")
	back.SetLabelColor(tcell.ColorTeal)
	back.SetSelectedFunc(func() {
		app.SetRoot(createMainMenu(app), true)
	})

	flex.AddItem(title, 1, 0, false)
	flex.AddItem(form, 0, 1, true)
	flex.AddItem(back, 1, 0, false)

	return flex
}

func createMainMenu(app *tview.Application) *tview.Flex {
	flex := tview.NewFlex().SetDirection(tview.FlexRow)

	header := tview.NewTextView().
		SetText("Buggy - Terminal User Interface").
		SetTextAlign(tview.AlignCenter).
		SetTextColor(tcell.ColorTeal)

	menu := tview.NewList().
		SetSelectedTextColor(tcell.ColorTeal).
		SetSelectedBackgroundColor(tcell.ColorDarkGray)

	menu.AddItem("Dashboard", "System metrics and status", rune('1'), func() {
		app.SetRoot(createDashboard(app), true)
	})
	menu.AddItem("File Manager", "Browse files and directories", rune('2'), func() {
		app.SetRoot(createFileManager(app), true)
	})
	menu.AddItem("Text Editor", "Simple text editor", rune('3'), func() {
		app.SetRoot(createTextEditor(app), true)
	})
	menu.AddItem("Monitor", "Real-time system monitor", rune('4'), func() {
		app.SetRoot(createMonitor(app), true)
	})
	menu.AddItem("Form Wizard", "Multi-step form", rune('5'), func() {
		app.SetRoot(createFormWizard(app), true)
	})
	menu.AddItem("Exit", "Close Buggy", rune('q'), func() {
		app.Stop()
	})

	menu.SetBorder(true).SetTitle(" Menu ")

	footer := tview.NewTextView().
		SetText("Navigate with arrow keys, Enter to select").
		SetTextAlign(tview.AlignCenter).
		SetTextColor(tcell.ColorGray)

	flex.AddItem(header, 3, 0, false)
	flex.AddItem(menu, 0, 1, true)
	flex.AddItem(footer, 1, 0, false)

	flex.SetBorderPadding(1, 1, 2, 2)

	return flex
}

func bar(percent int) string {
	filled := percent / 10
	empty := 10 - filled
	return "##########"[:filled] + "----------"[:empty]
}
