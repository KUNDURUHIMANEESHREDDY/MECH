export class LayoutManager {
  private activeLayout: string = 'default';

  getLayout(): string {
    return this.activeLayout;
  }

  setLayout(layout: string): void {
    this.activeLayout = layout;
  }
}

const layoutManager = new LayoutManager();

export { layoutManager };
