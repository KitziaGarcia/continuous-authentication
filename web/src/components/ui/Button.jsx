const VARIANT_CLASS = {
  primary: "button button-primary",
  quiet: "button button-quiet",
  text: "text-button",
  icon: "icon-button",
};

export default function Button({
  variant = "primary",
  full = false,
  className = "",
  type = "button",
  children,
  ...props
}) {
  const baseClass = VARIANT_CLASS[variant] || VARIANT_CLASS.primary;
  const classes = [baseClass, full && "button-full", className].filter(Boolean).join(" ");

  return (
    <button type={type} className={classes} {...props}>
      {children}
    </button>
  );
}
