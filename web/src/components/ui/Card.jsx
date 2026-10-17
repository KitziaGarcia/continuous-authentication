export default function Card({ as: Element = "div", className = "", children, ...props }) {
  return (
    <Element className={`panel ${className}`.trim()} {...props}>
      {children}
    </Element>
  );
}
