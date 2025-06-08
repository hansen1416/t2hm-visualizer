import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


class SimpleFileSelector:
    def __init__(self):
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "Select Result Folder", 400, 200
        )
        self._init_ui()
        gui.Application.instance.run()

    def _init_ui(self):
        em = self.window.theme.font_size
        layout = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))

        self.label = gui.Label("Select a results folder:")
        self.button = gui.Button("Browse")
        self.button.set_on_clicked(self._on_browse)

        layout.add_child(self.label)
        layout.add_child(self.button)

        self.window.add_child(layout)

    def _on_browse(self):
        dlg = gui.FileDialog(
            gui.FileDialog.OPEN_DIR, "Select Folder", self.window.theme
        )
        dlg.set_on_cancel(self._on_cancel)
        dlg.set_on_done(self._on_done)
        self.window.show_dialog(dlg)

    def _on_cancel(self):
        self.window.close_dialog()

    def _on_done(self, folder_path):
        self.window.close_dialog()
        self.window.close()

        # for video_name in os.listdir(folder_path):
        #     video_path = os.path.join(
        #         os.path.expanduser("~"), "Downloads", "videos", f"{video_name}.mp4"
        #     )
        #     if not os.path.exists(video_path):
        #         print(f"Video file does not exist: {video_path}")
        #         continue

        #     joints_glob = torch.load(
        #         os.path.join(folder_path, video_name, "joints_glob.pt")
        #     )
        #     verts_glob = torch.load(
        #         os.path.join(folder_path, video_name, "verts_glob.pt")
        #     )
        #     player = GVHMRPlayer(verts_glob.cpu().numpy(), video_path)
        #     player.play()
        #     break  # remove this if you want to loop through all videos


if __name__ == "__main__":
    SimpleFileSelector()
